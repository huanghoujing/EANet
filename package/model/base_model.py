import torch.nn as nn


class BaseModel(nn.Module):

    def get_ft_and_new_params(self, *args, **kwargs):
        """Get finetune and new parameters, mostly for creating optimizer.
        Return two lists."""
        return [], list(self.parameters())

    def get_ft_and_new_modules(self, *args, **kwargs):
        """Get finetune and new modules, mostly for setting train/eval mode.
        Return two lists."""
        return [], list(self.modules())

    def set_train_mode(self, *args, **kwargs):
        """Set model to train mode for model training, some layers can be fixed and set to eval mode."""
        self.train()
