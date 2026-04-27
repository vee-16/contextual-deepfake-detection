"""Stage 3 contextual reasoning module.

This module evaluates whether an image appears typical or atypical in
real-world context using a pretrained CLIP model.
"""



from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Union

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor


ImageInput = Union[Image.Image, str, Path]


class ContextReasoner:
    """Reason about whether an image is contextually typical or atypical.

    The class uses CLIP image-text similarity to compare an image against two
    prompt sets:
    - Typical prompts (normal real-world context)
    - Atypical prompts (unnatural or suspicious context)
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        threshold: float = 0.5,
    ) -> None:
        """Initialize the context reasoner and load model resources.

        Args:
            model_name: Hugging Face model id for a pretrained CLIP model.
            threshold: Decision threshold for atypical prediction. Values above
                threshold are labeled as "atypical".
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.typical_prompts: List[str] = [
            "a natural photo with anatomically correct human body proportions",
            "a realistic portrait with consistent facial symmetry and skin texture",
            "a real human hand with five fingers and natural finger joints",
            "a real-world scene with physically consistent lighting and shadows",
            "a candid photo with coherent depth, perspective, and occlusion",
            "a high-resolution photograph with natural fine-grained texture details",
            "a realistic image with clean boundaries around hair, hands, and clothing",
            "a photograph with readable and correctly formed text and logos",
            "a realistic face with aligned eyes, natural teeth, and consistent ears",
            "a natural image without duplicated objects or repeated background patterns",
        ]

        self.atypical_prompts: List[str] = [
            "a synthetic portrait with distorted facial anatomy or asymmetry artifacts",
            "an ai-generated image with malformed hands, extra fingers, or fused fingers",
            "an image with inconsistent limb proportions or impossible human pose",
            "a manipulated scene with mismatched lighting direction and unrealistic shadows",
            "an image with texture artifacts, waxy skin, or smeared fine details",
            "a fake image with broken edges around hair, fingers, or accessories",
            "an unnatural composition with inconsistent perspective or object geometry",
            "an ai image with garbled text, misspelled signs, or unreadable logos",
            "a generated face with misaligned eyes, warped teeth, or irregular earrings",
            "an image showing duplicated objects or repeating background textures",
        ]
        self._typical_text_emb: torch.Tensor | None = None
        self._atypical_text_emb: torch.Tensor | None = None

    def _extract_feature_tensor(self, output: object) -> torch.Tensor:
        """Normalize feature return types across transformers versions."""
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        if isinstance(output, (tuple, list)) and output:
            if isinstance(output[0], torch.Tensor):
                return output[0]
        raise TypeError("Model output does not contain a valid feature tensor.")

    @staticmethod
    def _normalize_context_score(typical_score: float, atypical_score: float) -> float:
        """Map score delta to [0, 1], where higher means more atypical."""
        delta = atypical_score - typical_score
        raw_score = (delta + 2.0) / 4.0
        return float(max(0.0, min(1.0, raw_score)))

    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load an image from a file path safely.

        Args:
            image_path: Path to the image file.

        Returns:
            A PIL image converted to RGB.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the file cannot be opened as a valid image.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image path does not exist: {path}")

        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Unable to load valid image from: {path}") from exc

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a PIL image into a normalized CLIP embedding.

        Args:
            image: PIL image to encode.

        Returns:
            A tensor of shape [1, embedding_dim] on the current device.

        Raises:
            TypeError: If the input is not a PIL image.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("encode_image expects a PIL.Image.Image input.")

        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_emb = self._extract_feature_tensor(
                self.model.get_image_features(**inputs)
            )

        image_emb = F.normalize(image_emb, p=2, dim=-1)
        return image_emb

    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        """Encode text prompts into normalized CLIP embeddings.

        Args:
            prompts: A list or sequence of prompt strings.

        Returns:
            A tensor of shape [num_prompts, embedding_dim] on current device.

        Raises:
            ValueError: If prompts are empty.
        """
        if not prompts:
            raise ValueError("encode_text received an empty prompts sequence.")

        inputs = self.processor(
            text=list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_emb = self._extract_feature_tensor(
                self.model.get_text_features(**inputs)
            )

        text_emb = F.normalize(text_emb, p=2, dim=-1)
        return text_emb

    def compute_similarity(
        self, image_emb: torch.Tensor, text_emb: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity between image and text embeddings.

        Args:
            image_emb: Image embedding tensor [1, dim] (or [N, dim]).
            text_emb: Text embedding tensor [M, dim].

        Returns:
            Similarity tensor of shape [N, M], where each value is cosine
            similarity in approximately [-1, 1].
        """
        return image_emb @ text_emb.T

    def _get_prompt_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and cache prompt embeddings for reuse across images."""
        if self._typical_text_emb is None:
            self._typical_text_emb = self.encode_text(self.typical_prompts)
        if self._atypical_text_emb is None:
            self._atypical_text_emb = self.encode_text(self.atypical_prompts)
        return self._typical_text_emb, self._atypical_text_emb

    def predict_label(self, context_score: float) -> str:
        """Predict context label from normalized context score.

        Args:
            context_score: Score in [0, 1], higher means more atypical.

        Returns:
            "atypical" if score >= threshold, otherwise "typical".
        """
        return "atypical" if context_score >= self.threshold else "typical"

    def score_image(self, image: ImageInput) -> Dict[str, Union[float, str]]:
        """Run full contextual reasoning pipeline for one image.

        Args:
            image: PIL image or image path.

        Returns:
            Dictionary with:
            - typical_score
            - atypical_score
            - context_score (0 to 1, higher means more atypical)
            - predicted_context_label ("typical" or "atypical")

        Raises:
            TypeError: If the image type is unsupported.
            ValueError/FileNotFoundError: For invalid image paths/files.
        """
        if isinstance(image, (str, Path)):
            pil_image = self.load_image(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise TypeError(
                "score_image expects a PIL.Image.Image, str path, or Path object."
            )

        image_emb = self.encode_image(pil_image)
        typical_text_emb, atypical_text_emb = self._get_prompt_embeddings()

        typical_sim = self.compute_similarity(image_emb, typical_text_emb).squeeze(0)
        atypical_sim = self.compute_similarity(image_emb, atypical_text_emb).squeeze(0)

        typical_score = float(typical_sim.mean().item())
        atypical_score = float(atypical_sim.mean().item())

        context_score = self._normalize_context_score(typical_score, atypical_score)

        predicted_context_label = self.predict_label(context_score)

        return {
            "typical_score": typical_score,
            "atypical_score": atypical_score,
            "context_score": context_score,
            "predicted_context_label": predicted_context_label,
        }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run contextual reasoning on a single image."
    )
    parser.add_argument(
        "--image",
        default="data/openfake/test/real/real_00000.png",
        help="Path to the image to score.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help='Decision threshold for "atypical" prediction (default: 0.5).',
    )
    parser.add_argument(
        "--model-name",
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model id to load.",
    )
    parser.add_argument(
        "--formatted",
        action="store_true",
        help="Pretty-print JSON output for easier reading.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    reasoner = ContextReasoner(model_name=args.model_name, threshold=args.threshold)

    try:
        result = reasoner.score_image(args.image)
        if args.pretty:
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(result))
    except Exception as exc:
        print(f"Failed to score image: {exc}")
