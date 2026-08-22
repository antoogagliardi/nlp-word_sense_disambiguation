import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score
from transformers import BertTokenizerFast, BertModel


class WSD_Model(pl.LightningModule):
	def __init__(self, fine_tuning:bool=False, token_path:str=None, model_path:str=None):
		super(WSD_Model, self).__init__()

		# TOKENIZER and BERT TRANSFORMER
		# self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
		# self.bert_transformer = BertModel.from_pretrained("bert-base-uncased")
		self.tokenizer = BertTokenizerFast.from_pretrained(token_path)
		self.bert_transformer = BertModel.from_pretrained(model_path)

		self.hidden_size = self.bert_transformer.config.hidden_size 					# Transformer Latent Dimention
		self.max_seq_length = self.bert_transformer.config.max_position_embeddings		# Max str len the Transformer can process (512)
		self.pad_token_id = self.tokenizer.pad_token_id 								# [PAD] Token ID
		self.cls_token_id = self.tokenizer.cls_token_id 								# [CLS] Token ID
		self.sep_token_id = self.tokenizer.sep_token_id 								# [SEP] Token ID

		if fine_tuning == False:
			# FREEZE The BERT Model
			print(f"Fine Tuning: {fine_tuning}\nFreezing the entire model")
			for _, param in self.bert_transformer.named_parameters():
				param.requires_grad_(False)
		else:
			# FREEZE The BERT Model
			print(f"Fine Tuning: {fine_tuning}")
			for _, param in self.bert_transformer.named_parameters():
				param.requires_grad_(False)
			encoder_defreeze = 4
			print(f"Defreeze Last {encoder_defreeze} encoder of the model for fine tuning")
			# Defreeze the BERT encoder from last one minus "encoder_defreeze" Parameter
			for name, param in self.bert_transformer.encoder.layer[-encoder_defreeze:].named_parameters():
				param.requires_grad = True

		# DROPOUT
		dropout_rate = 0.5
		self.dropout_layer = nn.Dropout(p=dropout_rate)

		# SENSE CLASSIFIER
		self.senses_classifier = nn.Linear(in_features=self.hidden_size, out_features=1, bias=True)

		# METRICS DEFINITION
		self.loss_function = nn.BCEWithLogitsLoss(reduction='mean')
		self.f1_score = BinaryF1Score(threshold=0.45, multidim_average='global')

	def forward(self, sentences):
		# Get the element of Sentences for the the transformer
		# NB: remember that BERT want element to be squeezed before passing into the encoder
		input_ids = sentences["input_ids"].squeeze(1)
		input_attention_mask = sentences["attention_mask"].squeeze(1)
		input_token_type_ids = sentences["token_type_ids"].squeeze(1)

		# Compute the output with the Transformer (last_hidden_state, pooler_output, hidden_states)
		output = self.bert_transformer(input_ids= input_ids,
										attention_mask = input_attention_mask,
										token_type_ids = input_token_type_ids,
										output_hidden_states=True)
		# Take the [CLS] Embedding of the last hidden state
		cls_last_hidden = output["pooler_output"]

		features= self.dropout_layer(cls_last_hidden)
		features = self.senses_classifier(cls_last_hidden)

		return features

	def configure_optimizers(self):
		learning_rate = 2e-5
		optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		context_gloss_sentences = train_batch[0]
		true_output = train_batch[1]
		true_output = true_output.unsqueeze(1)

		features = self(sentences=context_gloss_sentences)

		loss = self.loss_function(features, true_output)	# Compute the training Loss Function
		train_f1 = self.f1_score(features, true_output)		# Compute the training F1 Score

		self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
		self.log('train_F1score', train_f1, on_step=True, on_epoch=True, prog_bar=True)

		return {"loss": loss, "train_f1": train_f1}

	def validation_step(self, val_batch, batch_idx):
		context_gloss_sentences = val_batch[0]
		true_output = val_batch[1]
		true_output = true_output.unsqueeze(1)

		features = self(sentences=context_gloss_sentences)

		loss = self.loss_function(features, true_output)	# Compute the validation Loss Function
		val_f1 = self.f1_score(features, true_output) 		# Compute the validation F1 Score
		
		self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
		self.log('valid_F1score', val_f1, on_step=True, on_epoch=True, prog_bar=True)

		return {"loss": loss, "val_f1": val_f1}