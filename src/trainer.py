import torch
from torch import nn
from transformers import Trainer
from transformers.utils import is_apex_available
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.modeling_utils import unwrap_model
if is_apex_available():
    from apex import amp

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            model.eval()
            if type(model) is torch.nn.DataParallel:
                generated_ids = model.module.generate(inputs['input_ids'][:,:-1].to(model.module.device), attention_mask = inputs['attention_mask'][:,:-1].to(model.module.device), max_length = inputs['input_ids'].shape[-1], pad_token_id=self.tokenizer.eos_token_id)
            else:
                generated_ids = model.generate(inputs['input_ids'][:,:-1].to(model.device), attention_mask = inputs['attention_mask'][:,:-1].to(model.device), max_length = inputs['input_ids'].shape[-1], pad_token_id=self.tokenizer.eos_token_id)
        model.train()
        inputs['input_ids'] = torch.cat([inputs['input_ids'], generated_ids], dim = 0)
        inputs['attention_mask'] = torch.cat([inputs['attention_mask'].to(model.module.device), inputs['attention_mask'].to(model.module.device)], dim = 0)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()