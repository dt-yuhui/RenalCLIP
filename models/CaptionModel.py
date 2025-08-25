import torch
import torch.nn as nn

class ImageCaptionModel(nn.Module):
    """
    Building a LLaVA-like Multimodal Model:
    1. The image is processed by a frozen, pre-trained image encoder to get a feature vector.
    2. The feature vector is mapped to a fixed-length token sequence [num_image_tokens x hidden_dim] using a projection layer.
    3. The resulting image tokens are concatenated with the text tokens (from a caption tokenizer)
    and fed into a language model for text generation and loss calculation.
    The loss is not calculated for the prefix part.
    """
    def __init__(self, 
                 image_encoder, 
                 language_model, 
                 prefix_tokens, 
                 suffix_tokens, 
                 projection_hidden_dim, 
                 num_image_tokens,
                 image_feature_dim):
        """
        image_encoder: pre-trained image ecndoer
        language_model: Pre-trained language model (e.g., BioMistral-7B)
        projection_hidden_dim: hidden size of LLM
        num_image_tokens: How many "virtual" tokens to expand the image features into
        image_feature_dim: feature_dim output by image encoder
        """
        super().__init__()
        self.image_encoder = image_encoder
        # freeze image encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.num_image_tokens = num_image_tokens
        self.language_model = language_model
        self.prefix_tokens = prefix_tokens
        self.suffix_tokens = suffix_tokens


        self.projection_hidden_dim = projection_hidden_dim
        # Map the image feature to a vector of (num_image_tokens * hidden_dim)
        # then reshape it to [num_image_tokens, hidden_dim]
        self.projection = nn.Linear(image_feature_dim, num_image_tokens * projection_hidden_dim, dtype=torch.bfloat16)
        
        self.prefix_len = self.prefix_tokens.size(1)
        self.suffix_len = self.suffix_tokens.size(1)
    
    def forward(self, images, input_ids, attention_mask=None):
        """
        images: Tensor, shape [batch_size, channels, height, width]
        input_ids: Tensor, shape [batch_size, seq_len], tokenized ids of caption
        attention_mask: Tensor, shape [batch_size, seq_len] (optional)
        """
        batch_size = images.size(0)
        # image features, shape: [batch_size, feature_dim]
        image_features = self.image_encoder(images).bfloat16()
        # Project to obtain the image's "virtual" token representation
        projected = self.projection(image_features)  # [batch_size, num_image_tokens * hidden_dim]
        projected = projected.view(batch_size, self.num_image_tokens, self.projection_hidden_dim)  # [batch, num_image_tokens, hidden_dim]
        
        prefix_tokens = self.prefix_tokens.to(images.device)
        suffix_tokens = self.suffix_tokens.to(images.device)
        
        prefix_embeds = self.language_model.get_input_embeddings()(prefix_tokens)
        suffix_embeds = self.language_model.get_input_embeddings()(suffix_tokens)
        
        prefix_embeds = prefix_embeds.expand(batch_size, -1, -1)
        suffix_embeds = suffix_embeds.expand(batch_size, -1, -1)
        
        # Use the language model's embedding layer to convert tokens into embeddings.
        # It's recommended to use the get_input_embeddings() method here, 
        # though note that the exact method name may vary for different models.
        text_embeds = self.language_model.get_input_embeddings()(input_ids)  # [batch, seq_len, hidden_dim]
        
        # Combine these in the following order: 
        #   prefix_embeds, image tokens, suffix_embeds, original caption embedding.
        # combined_embeds = torch.cat([projected, text_embeds], dim=1)  # [batch, num_image_tokens+seq_len, hidden_dim]
        combined_embeds = torch.cat([
            prefix_embeds,       # <s> [INST] 
            projected,           # <image tokens>
            suffix_embeds,       # prompt
            text_embeds          # original caption
        ], dim=1)
                
        prefix_mask = torch.ones(batch_size, self.prefix_len, device=images.device) 
        img_mask = torch.ones(batch_size, self.num_image_tokens, device=images.device)
        suffix_mask = torch.ones(batch_size, self.suffix_len, device=images.device)
        
        if attention_mask is None:
            text_mask = torch.ones(input_ids.size(), device=images.device, dtype=torch.long)
        else:
            text_mask = attention_mask
            
        combined_attention_mask = torch.cat([prefix_mask, img_mask, suffix_mask, text_mask], dim=1)
        
        # Both the instruction and image tokens are masked with -100 and will not contribute to the loss
        prefix_labels = torch.full((batch_size, self.prefix_len), -100, device=images.device, dtype=input_ids.dtype)
        img_labels = torch.full((batch_size, self.num_image_tokens), -100, device=images.device, dtype=input_ids.dtype)
        suffix_labels = torch.full((batch_size, self.suffix_len), -100, device=images.device, dtype=input_ids.dtype)
        if attention_mask is None:
            text_labels = input_ids
        else:
            text_labels = input_ids.masked_fill(attention_mask == 0, -100)
        
        combined_labels = torch.cat([prefix_labels, img_labels, suffix_labels, text_labels], dim=1)
        
        outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=combined_labels
        )
        return outputs
    
    def generate(self, images, **generate_kwargs):

        batch_size = images.size(0)
        
        image_features = self.image_encoder(images).bfloat16()
        projected = self.projection(image_features)
        projected = projected.view(batch_size, self.num_image_tokens, self.projection_hidden_dim)
        
        prefix_tokens = self.prefix_tokens.to(images.device)
        suffix_tokens = self.suffix_tokens.to(images.device)
        
        prefix_embeds = self.language_model.get_input_embeddings()(prefix_tokens)
        suffix_embeds = self.language_model.get_input_embeddings()(suffix_tokens)
        
        prefix_embeds = prefix_embeds.expand(batch_size, -1, -1)
        suffix_embeds = suffix_embeds.expand(batch_size, -1, -1)
        
        prompt_embeds = torch.cat([
            prefix_embeds,       # [INST] 
            projected,           # <image tokens>
            suffix_embeds        # prompt
        ], dim=1)
        
        total_seq_length = prompt_embeds.size(1)
        attention_mask = torch.ones((batch_size, total_seq_length), device=images.device, dtype=torch.long)
        
        # Set the pad_token_id (if not specified in generate_kwargs)
        if 'pad_token_id' not in generate_kwargs:
            generate_kwargs['pad_token_id'] = self.language_model.config.eos_token_id
        
        generated_ids = self.language_model.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=attention_mask,
            **generate_kwargs
        )
        
        return generated_ids