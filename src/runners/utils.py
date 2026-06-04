import torch

class LossScaler:
    """
    Lightweight AMP loss-scaler.
    """

    state_dict_key = "amp_scaler"

    def __init__(self, use_amp: bool = True, device: str = "cuda"):
        self._scaler = torch.amp.GradScaler(device, enabled=use_amp) # type:ignore

    def __call__(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        clip_grad: float | None = None,
        parameters=None,
        update_grad: bool = True,
    ) -> None:
        self._scaler.scale(loss).backward()
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None, "clip_grad needs parameters"
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self) -> dict:
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self._scaler.load_state_dict(state_dict)