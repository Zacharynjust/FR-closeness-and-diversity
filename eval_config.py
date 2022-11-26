configurations = {
    'eval': dict(
        EVAL_LIST = ['lfw', 'cfp_fp', 'agedb', 'ijbb', 'ijbc', 'megaface'], # support: 'lfw', 'cfp_fp', 'agedb', 'ijbb', 'ijbc', 'megaface'
        MODEL_PATH = 'pretrained/ours_sms_r100.pt', 
        MODEL_TYPE = 'IR_100',
        LOG_ROOT = './runs/eval_result/',
        BASE_ROOT = "eval_data/",
        IJB_ROOT = "/datasets/face/IJB_release/",
        MEGA_ROOT = "/datasets/face/megaface_testpack/",
        GPU = 0
    ),
}