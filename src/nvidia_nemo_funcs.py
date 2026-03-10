from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf
import json
import os

def create_config(audio_path, output_dir, speakers):
    config = OmegaConf.load("./config_files/diar_infer_general.yaml")
 
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
 
    meta = {
        "audio_filepath": audio_path,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": None,
        "uem_filepath": None,
        "num_speakers": speakers, 

    }
    with open(os.path.join(data_dir, "input_manifest.json"), "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")

    
    pretrained_vad = "vad_multilingual_marblenet"
    config.diarizer.vad.model_path = pretrained_vad

    pretrained_speaker_model = "titanet_large"
    config.num_workers = 0  # Workaround for multiprocessing hanging with ipython issue
    config.diarizer.manifest_filepath = os.path.join(data_dir, "input_manifest.json")
    config.diarizer.out_dir = (
        output_dir  # Directory to store intermediate files and prediction outputs
    )
    
    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.oracle_vad = (
        False  # compute VAD provided with model_path to vad config
    )
    config.diarizer.clustering.parameters.oracle_num_speakers = speakers

    #config.device = "cuda"

    return config


def diarization(audio_path, speakers):
    cfg = create_config(audio_path, "./nemo_out", speakers)
    sd_model = ClusteringDiarizer(cfg=cfg)
    sd_model.diarize()

    

