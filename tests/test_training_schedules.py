import unittest

from tfts.training import exposure_bias
from tfts.training.schedules import annealed_noise_std, teacher_forcing_decay


class TrainingSchedulesTest(unittest.TestCase):
    def test_epoch_schedules_keep_warmup_and_clamp_at_end(self):
        for epoch, teacher, noise in ((1, 1.0, 0.05), (3, 1.0, 0.05), (5, 0.6, 0.275), (7, 0.2, 0.5), (10, 0.2, 0.5)):
            with self.subTest(epoch=epoch):
                self.assertAlmostEqual(teacher_forcing_decay(epoch, total_epochs=7), teacher)
                self.assertAlmostEqual(annealed_noise_std(epoch, total_epochs=7), noise)

    def test_exposure_bias_reuses_canonical_schedule(self):
        self.assertIs(exposure_bias.annealed_noise_std, annealed_noise_std)
