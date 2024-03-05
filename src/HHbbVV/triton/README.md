## Triton-server

See READMEs in model directories for links to models and/or configs.

To export to onnx:

```bash
modeldir="../../hhbbww/HHbbVV/src/HHbbVV/triton/models/model_2023May30/ak8_MD_inclv8_part_2reg_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3/"

# based on this command https://github.com/colizz/weaver-core-dev/blob/d038cd502d1b4a8ab3deefa9c3022bd3b812baf5/cmd_stage2.md?plain=1#L979-L1010
python weaver/train.py --train-mode hybrid \
-o loss_gamma 5 -o fc_params '[(512,0.1)]' -o embed_dims '[64,256,64]' -o pair_embed_dims '[32,32,32]' --use-amp \
--data-config weaver/data_new/inclv7plus/ak8_MD_inclv8_part_2reg_manual.yaml \
--network-config weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $modeldir/net_best_epoch_state.pt \
--export-onnx $modeldir/model.onnx
```

For the 2023May30 model I had to:

1. Bypass the if statement here
   https://github.com/colizz/weaver-core-dev/blob/d038cd502d1b4a8ab3deefa9c3022bd3b812baf5/weaver/networks/ParticleTransformer2023.py#L586
   (just `return output` instead) - torch complained about comparing a tensor to
   a Python boolean.
2. Remove the softmax part
   https://github.com/colizz/weaver-core-dev/blob/d038cd502d1b4a8ab3deefa9c3022bd3b812baf5/weaver/networks/ParticleTransformer2023.py#L583-L584
   (doesn't make sense since regression outputs are included).
3. Rewrite this einsum statement
   https://github.com/colizz/weaver-core-dev/blob/d038cd502d1b4a8ab3deefa9c3022bd3b812baf5/weaver/networks/ParticleTransformer2023.py#L448C14-L448C14
   ->
   https://github.com/colizz/weaver-core-dev/blob/d038cd502d1b4a8ab3deefa9c3022bd3b812baf5/weaver/networks/example_ParticleTransformerTagger_hybrid_outputWithHidNeurons.py#L337 -
   onnx opset v11 doesn't support einsum.
