import wespeaker.models.ecapa_tdnn as ecapa_tdnn
import wespeaker.models.resnet as resnet
import wespeaker.models.samresnet as samresnet


def get_speaker_model(model_name: str):
    if model_name.startswith("ECAPA_TDNN"):
        return getattr(ecapa_tdnn, model_name)
    elif model_name.startswith("ResNet"):
        return getattr(resnet, model_name)
    elif model_name.startswith("SimAM_ResNet"):
        return getattr(samresnet, model_name)
    else:
        print(model_name + " not found !!!")
        exit(1)
