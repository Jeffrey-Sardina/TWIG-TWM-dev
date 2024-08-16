from twig_twm import do_app_job

if __name__ == '__main__':
    hyps_dict_path = "output/CoDExSmall/CoDExSmall-ComplEx-TWM-run2.1/CoDExSmall-TWM-run2.1.grid"
    kge_model_name = 'ComplEx'
    run_id = '2.1'
    model_save_path = "checkpoints/chkpt-ID_8606280476482954_tag_TWIG-job_UMLS_e1-e0.pt"
    model_config_path = "checkpoints/chkpt-ID_8606280476482954_tag_TWIG-job_UMLS.pkl"
    do_app_job(
        hyps_dict_path=hyps_dict_path,
        kge_model_name=kge_model_name,
        run_id=run_id,
        model_save_path=model_save_path,
        model_config_path=model_config_path
    )
