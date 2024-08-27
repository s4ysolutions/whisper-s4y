from typing import Union

from transformers import TFWhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer, \
    TFForceTokensLogitsProcessor, TFSuppressTokensLogitsProcessor, TFSuppressTokensAtBeginLogitsProcessor
from transformers.models.whisper.modeling_tf_whisper import TFWhisperEncoder, TFWhisperDecoder, TFWhisperMainLayer
from whisper_s4y.whisper.huggingface import tf_logits_process_fix as fix

TFForceTokensLogitsProcessor.__init__ = fix.TFForceTokensLogitsProcessor_patched__init__
TFForceTokensLogitsProcessor.__call__ = fix.TFForceTokensLogitsProcessor_patched__call__
TFSuppressTokensLogitsProcessor.__call__ = fix.TFSuppressTokensLogitsProcessor_patched__call__
TFSuppressTokensAtBeginLogitsProcessor.__call__ = fix.TFSuppressTokensAtBeginLogitsProcessor_patched__call__


def for_conditional_generation(model_id: str) -> TFWhisperForConditionalGeneration:
    return TFWhisperForConditionalGeneration.from_pretrained(model_id, from_pt=True)


def main_layer(model: Union[TFWhisperForConditionalGeneration, str]) -> TFWhisperMainLayer:
    if isinstance(model, str):
        model = for_conditional_generation(model)
    return model.model


def encoder(model: Union[TFWhisperForConditionalGeneration, str]) -> TFWhisperEncoder:
    if isinstance(model, str):
        model = for_conditional_generation(model)
    return model.model.encoder


def decoder(model: [TFWhisperForConditionalGeneration, str]) -> TFWhisperDecoder:
    if isinstance(model, str):
        model = for_conditional_generation(model)
    return model.model.decoder


def processor(model_id: str) -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(model_id)


def feature_extractor(processorOrId: Union[WhisperProcessor, str]) -> WhisperFeatureExtractor:
    if isinstance(processorOrId, str):
        _processor = processor(processorOrId)
    else:
        _processor = processorOrId
    return _processor.feature_extractor


def tokenizer(processorOrId: Union[WhisperProcessor, str]) -> WhisperTokenizer:
    if isinstance(processorOrId, str):
        _processor = processor(processorOrId)
    else:
        _processor = processorOrId
    return _processor.tokenizer
