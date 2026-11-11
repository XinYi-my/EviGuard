"""Tiny randomly initialized HF LLaVA, never a benchmark or pretrained model."""
import torch
from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration
from peft import LoraConfig,get_peft_model,TaskType
from src.models.eviguard_model import EviGuardModel
from src.models.llava_utils import language_lora_targets
from src.models.checkpoint import save_checkpoint,restore_checkpoint

class EvidenceTokenizer:
    def __call__(self,strings,**kwargs):
        return {"input_ids":torch.ones(len(strings),2,dtype=torch.long),"attention_mask":torch.ones(len(strings),2,dtype=torch.long)}

def test_real_hf_llava_peft_gradient_and_complete_roundtrip(tmp_path):
    vision=CLIPVisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,image_size=8,patch_size=4)
    text=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,pad_token_id=0)
    config=LlavaConfig(vision_config=vision.to_dict(),text_config=text.to_dict(),image_token_index=31,vision_feature_layer=-1,image_seq_length=4)
    base=LlavaForConditionalGeneration(config)
    model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=language_lora_targets(base,["q_proj","v_proj"]),task_type=TaskType.CAUSAL_LM))
    for name,parameter in model.named_parameters():
        if "multi_modal_projector" in name:parameter.requires_grad=True
        if "vision_tower" in name:assert not parameter.requires_grad
    cfg={"model":{"name":"random-tiny-llava-test","arch":"llava"},"rsf":{"bottleneck_dim":4,"dropout":0.,"gate_hidden_dim":2,"evidence_max_tokens":2,"default_reliability":.5}}
    eg=EviGuardModel(model,None,None,EvidenceTokenizer(),cfg).eval()
    inputs={"input_ids":torch.tensor([[1,31,31,31,31,2,3]]),"attention_mask":torch.ones(1,7,dtype=torch.long),"pixel_values":torch.randn(1,3,8,8)}
    out=eg.forward_with_rsf(["test"],torch.tensor([.7]),**inputs)
    assert out["logits"].shape==(1,7,32)
    out["logits"][:,-1].square().sum().backward()
    assert eg.connector.linear_1.weight.grad.abs().sum()>0
    assert eg.rsf.W_delta.weight.grad.abs().sum()>0
    expected=out["logits"].detach()
    save_checkpoint(eg,tmp_path)
    with torch.no_grad():eg.connector.linear_1.weight.add_(1)
    restore_checkpoint(eg,tmp_path)
    actual=eg.forward_with_rsf(["test"],torch.tensor([.7]),**inputs)["logits"]
    assert torch.equal(expected,actual)
