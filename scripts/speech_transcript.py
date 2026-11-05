from espnet2.bin.s2t_inference import Speech2Text
# OWSM-CTC v4: multilingual ASR, translation and language ID in one model
s2t = Speech2Text.from_pretrained("espnet/owsm_ctc_v4_1B")
for start, end, text in s2t.decode_long("data/generated/speech/mixture_mic_all_enhance.wav"):  # any length or rate
    print(text)
