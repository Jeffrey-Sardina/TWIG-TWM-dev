from twm_app import start_twm_app

if __name__ == '__main__':
    hyps_dict_path = "output/CoDExSmall/CoDExSmall-ComplEx-TWM-run2.1/CoDExSmall-TWM-run2.1.grid"
    kge_model_name = 'ComplEx'
    run_id = '2.1'
    model_save_path = "checkpoints/chkpt-ID_8606280476482954_tag_TWIG-job_UMLS_e1-e0.pt"
    model_config_path = "checkpoints/chkpt-ID_8606280476482954_tag_TWIG-job_UMLS.pkl"
    start_twm_app(
        hyps_dict_path=hyps_dict_path,
        kge_model_name_local=kge_model_name,
        run_id_local=run_id,
        model_save_path_local=model_save_path,
        model_config_path_local=model_config_path
    )
