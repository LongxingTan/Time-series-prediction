"""Build-time resolution of native decoding and parallel teacher forcing."""

from .decoding import decode


class DecodePolicy:
    def __init__(self, backbone, capabilities, config):
        self.backbone = backbone
        self.config = config
        self.parallel = backbone.decode_teacher_forced if capabilities.supports_parallel_teacher_forcing else None

    def run(self, batch, *, training, probability):
        if (
            self.parallel is not None
            and isinstance(probability, (float, int))
            and probability == 1.0
            and batch.future_observed_mask is None
        ):
            return self.parallel(batch, training=training)
        return decode(
            self.backbone,
            batch,
            self.config.prediction_length,
            training=training,
            teacher_probability=probability,
            sampler=self.config.feedback_sampler if training else "point",
        )
