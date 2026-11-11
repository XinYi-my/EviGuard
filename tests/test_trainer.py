import torch
from types import SimpleNamespace
from src.training.trainer import EviGuardTrainer
from src.utils.helpers import load_config

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model=torch.nn.Linear(2,2)
        self.ref_model=torch.nn.Linear(2,2)
        self.rsf=torch.nn.Linear(2,1)
        self.arch="llava"
        self.processor=self.tokenizer=None
    def get_trainable_params(self):
        return list(self.model.parameters())+list(self.rsf.parameters())
    def forward_with_rsf(self,evidence_strings,reliabilities,**kw):
        x=torch.ones(len(evidence_strings),2)
        h=self.model(x)
        rs=torch.sigmoid(self.rsf(h.detach())).squeeze(-1)
        logits=h.unsqueeze(1).repeat(1,3,1)
        return {"logits":logits,"H_original":h.unsqueeze(1),"H_fused":h.unsqueeze(1),"risk_scores":rs}
    def forward_reference(self,**kw):
        return {"connector_activation":torch.zeros(1,1,2)}

def config(tmp_path):
    c=load_config("configs/safeeraser_llava.yaml")
    c["training"].update(mixed_precision="no",gradient_accumulation_steps=2,num_workers=0,warmup_steps=0)
    c["subspace"].update(rank_k=1,min_fit_samples=2)
    c["paths"]={k:str(tmp_path/k) for k in c["paths"]}
    return c

def test_stage1_updates_parameters_and_counts_optimizer_steps(tmp_path,monkeypatch):
    import src.data.datasets as data
    inp={"input_ids":torch.ones(1,3,dtype=torch.long),"attention_mask":torch.ones(1,3),"labels":torch.tensor([[-100,0,0]]),"forget_mask":torch.tensor([True]),"retain_mask":torch.tensor([False]),"pd_mask":torch.tensor([False]),"pixel_values":torch.ones(1,1,2),"evidence_strings":["evidence"],"reliabilities":torch.tensor([0.8])}
    monkeypatch.setattr(data,"prepare_model_inputs",lambda *a,**k:inp)
    model=Toy();c=config(tmp_path)
    trainer=EviGuardTrainer(model,[{}],cfg=c)
    before=model.model.weight.detach().clone()
    info=trainer._optimizer_step(iter([{},{}]),stage=1)
    assert not torch.equal(before,model.model.weight)
    assert info["total_loss"]>0
    assert trainer.optimizer_steps==1
    assert trainer.scheduler.last_epoch==1

def test_invalid_stage_budget_fails_before_training(tmp_path):
    import pytest
    c=config(tmp_path);c["training"]["total_steps"]=1
    with pytest.raises(ValueError,match="total_steps"):
        EviGuardTrainer(Toy(),[{}],cfg=c)


def test_training_keeps_full_stage2_budget_and_feedback_boundaries(tmp_path,monkeypatch):
    c=config(tmp_path)
    c["training"].update(stage1_steps=1,stage2_steps=5,total_steps=6,num_feedback_iterations=3,save_every_steps=100)
    trainer=EviGuardTrainer(Toy(),[{}],cfg=c)
    stages=[];fits=[];saved=[]
    def step(iterator,stage):
        stages.append(stage)
        trainer.optimizer_steps+=1
        trainer._iterator=iterator
        return {"total_loss":1.}
    def fit(hard=False):
        fits.append((trainer.optimizer_steps,hard))
        trainer.risk_subspace.is_fitted=True
    monkeypatch.setattr(trainer,"_optimizer_step",step)
    monkeypatch.setattr(trainer,"_fit_subspace",fit)
    monkeypatch.setattr(trainer,"_save",lambda path:saved.append(path.name))
    trainer.train()
    assert stages==[1,2,2,2,2,2]
    assert fits==[(1,False),(2,True),(4,True)]
    assert saved==["final"]
