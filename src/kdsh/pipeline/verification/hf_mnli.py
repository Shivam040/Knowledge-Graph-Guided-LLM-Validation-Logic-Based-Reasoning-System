from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class NLIResult:
    """Normalized NLI result."""
    label: str  # entailment | contradiction | neutral
    confidence: float
    probs: Dict[str, float]


class HFNLIVerifier:
    """HuggingFace MNLI verifier with batching + robust label mapping."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                "Missing MNLI dependencies. Install with: pip install -U transformers torch"
            ) from e

        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()

        # Device selection
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        self._model.to(self._device)

        # Label mapping
        self._ent_id, self._con_id, self._neu_id = self._infer_label_ids()

    def _infer_label_ids(self) -> Tuple[int, int, int]:
        """Infer label ids from model config; fallback to common MNLI ordering."""
        cfg = getattr(self._model, "config", None)
        id2label = {}
        if cfg is not None and hasattr(cfg, "id2label") and isinstance(cfg.id2label, dict):
            id2label = {int(i): str(l).lower() for i, l in cfg.id2label.items()}

        ent_id = con_id = neu_id = None
        for i, l in id2label.items():
            if "entail" in l:
                ent_id = i
            elif "contr" in l:
                con_id = i
            elif "neutral" in l:
                neu_id = i

        # Common MNLI: 0=contradiction, 1=neutral, 2=entailment
        if ent_id is None or con_id is None or neu_id is None:
            ent_id, con_id, neu_id = 2, 0, 1

        return ent_id, con_id, neu_id

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        return self.predict_batch([premise], [hypothesis])[0]

    def predict_batch(self, premises: List[str], hypotheses: List[str]) -> List[NLIResult]:
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must have same length")

        torch = self._torch
        out: List[NLIResult] = []

        for i in range(0, len(premises), self.batch_size):
            p = premises[i : i + self.batch_size]
            h = hypotheses[i : i + self.batch_size]

            enc = self._tokenizer(
                p,
                h,
                truncation="only_first",
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}

            with torch.no_grad():
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

            for row in probs:
                p_ent = float(row[self._ent_id])
                p_con = float(row[self._con_id])
                p_neu = float(row[self._neu_id])

                # predicted label by max prob
                if p_ent >= p_con and p_ent >= p_neu:
                    label = "entailment"
                    conf = p_ent
                elif p_con >= p_ent and p_con >= p_neu:
                    label = "contradiction"
                    conf = p_con
                else:
                    label = "neutral"
                    conf = p_neu

                out.append(
                    NLIResult(
                        label=label,
                        confidence=float(conf),
                        probs={"entailment": p_ent, "contradiction": p_con, "neutral": p_neu},
                    )
                )

        return out
