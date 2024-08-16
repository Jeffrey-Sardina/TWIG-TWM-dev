# TWIG imports
from load_data import _do_load
from twig_twm import load_config
from utils import load_custom_dataset, get_triples, calc_graph_stats

# external imports
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pykeen import datasets
import torch
import ast

device = 'cuda'

def load_twig_fmt_data(dataset_name, kge_model_name, run_id, normalisation, n_bins):
    # this is embarrassing but it's needed
    # bc TWIG load_data uses a hardcodeed a local relative path
    # so for now this work-around it ok
    datasets_to_load = {
        dataset_name: run_id
    }
    twig_data = _do_load(
        datasets_to_load=datasets_to_load,
        model_name=kge_model_name,
        test_ratio=0.0,
        valid_ratio=0.0,
        normalisation=normalisation,
        n_bins=n_bins,
        do_print=False
    )
    return twig_data

def invert_dict(dict_kv):
    dict_vk = {}
    for key in dict_kv:
        val = dict_kv[key]
        assert not val in dict_vk, f'This function assumes a structly 1:1 mapping, but that is not true for {key, val} in the original'
        dict_vk[val] = key
    return dict_vk

def create_TWM_graph(
        dataset_name,
        run_id,
        twig_data,
        exp_id_wanted,
        TWIG_model=None,
        do_print=True
    ):
    # load data
    try:
        dataset = datasets.get_dataset(dataset=dataset_name)
    except:
        dataset = load_custom_dataset(dataset_name)
    triples_dicts = get_triples(dataset)
    id_to_ent = invert_dict(dataset.entity_to_id)
    id_to_rel = invert_dict(dataset.relation_to_id)

    R_preds = None
    mrr_pred = None

    if TWIG_model:
        TWIG_model.eval()
    
    struct_tensor, hyps_tensor, mrr_true, rank_dist_true = twig_data.get_batch(
            dataset_name=dataset_name,
            run_id=run_id,
            exp_id=exp_id_wanted,
            mode='train'
        )
    rank_list_pred = TWIG_model(struct_tensor, hyps_tensor)
    rank_list_true = twig_data.rank_lists[dataset_name][run_id]['train'][exp_id_wanted]

    max_possible_rank = twig_data.max_ranks[dataset_name]
    if TWIG_model is not None:
        mrr_pred = torch.mean(1 / (1 + rank_list_pred * (max_possible_rank - 1)))
    else:
        mrr_true = torch.mean(1 / (rank_list_true * max_possible_rank)) # mrr from ranks on range [0,max]
    assert len(rank_list_pred) % 2 == 0, "should be a multiple of 2 for s and o corruption ranks"
    assert len(rank_list_true) % 2 == 0, "should be a multiple of 2 for s and o corruption ranks"

    # calculate all learnabilities
    learnabilities_pred = []
    learnabilities_true = []
    for i in range(len(R_preds) // 2):
        s_idx = i
        o_idx = (len(R_preds) // 2) + i

        # get predictions
        learnability_pred_s = 1 / (1 + rank_list_pred[s_idx] * (max_possible_rank - 1))
        learnability_pred_o = 1 / (1 + rank_list_pred[o_idx] * (max_possible_rank - 1))
        learnability_pred = (learnability_pred_s + learnability_pred_o) / 2
        assert learnability_pred == learnability_pred, 'Learnability should not be NaN!'
        learnability_pred = round(float(learnability_pred), 2)
        learnabilities_pred.append(learnability_pred)

        # get ground truth
        learnability_true_s = 1 / (rank_list_true[s_idx] * max_possible_rank)
        learnability_true_o = 1 / (rank_list_true[o_idx] * max_possible_rank)
        learnability_true = (learnability_true_s + learnability_true_o) / 2
        assert learnability_true == learnability_true, 'Learnability should not be NaN!'
        learnability_true = round(float(learnability_true), 2)
        learnabilities_true.append(learnability_true)

    assert len(learnabilities_pred) * 2 == len(R_preds), f"new version should have exactly half the size -- at the  triple level not subj and obj level. But itr was {len(learnabilities)}, len{len(R_preds)}"
    assert len(learnabilities_true) * 2 == len(R_preds), f"new version should have exactly half the size -- at the  triple level not subj and obj level. But itr was {len(learnabilities)}, len{len(R_preds)}"

    '''
    TODO: make sure triple order herer is the same as in ranks lists in the code above!!!
    '''
    graph_stats = calc_graph_stats(triples_dicts, do_print=False)
    twm = nx.MultiDiGraph()
    for triple_id, triple in enumerate(triples_dicts['train']):
        s, p, o = triple

        # get learnabilities
        learnability_pred = learnabilities_pred[triple_id]
        learnability_true = learnabilities_true[triple_id]

        # get pred data
        pred_freq = graph_stats['train']['pred_freqs'][p]

        # get node data
        s_deg = graph_stats['train']['degrees'][s]
        s_deg = round(float(s_deg), 2)
        o_deg = graph_stats['train']['degrees'][o]
        o_deg = round(float(o_deg), 2)

        # annotate graph with that data
        s_node = id_to_ent[s]
        o_node = id_to_ent[o]
        p_rel = id_to_rel[p]
        twm.add_node(s_node, name=s_node, pyKEEN_ID=s, train_deg=s_deg)
        twm.add_node(o_node, name=o_node, pyKEEN_ID=o, train_deg=o_deg)
        twm.add_edge(
            s_node,
            o_node,
            label=p_rel,
            name=p_rel,
            learnability_pred=learnability_pred,
            learnability_true=learnability_true,
            p_train_freq=pred_freq,
            pyKEEN_ID=p
        )
    return twm, mrr_pred, mrr_true

def draw_TWM_graph(twm, out_file_name, draw_pred):
    if draw_pred:
        edges, learnabilities = zip(
            *nx.get_edge_attributes(twm, 'learnability_pred').items()
        )
    else:
        edges, learnabilities = zip(
            *nx.get_edge_attributes(twm, 'learnability_true').items()
        )

    colours = []
    for learnability in learnabilities:
        colour = cm.YlOrBr(learnability)
        colours.append(colour)

    fig, ax = plt.subplots()
    node_pos = nx.circular_layout(twm)
    nx.draw(
        twm,
        node_size=1,
        with_labels=False,
        edgelist=edges,
        edge_color=colours,
        pos=node_pos
    )
    sm = plt.cm.ScalarMappable(cmap=cm.YlOrBr)
    sm._A = []
    plt.colorbar(sm, ax=ax)
    ax.axis('off')
    fig.set_facecolor('grey')
    plt.savefig(out_file_name)

def load_hyps_dict(path):
    with open(path, 'r') as inp:
        hps_dict = {}
        for line in inp:
            exp_id, hps_literal = line.strip().split(' --> ')
            exp_id = int(exp_id)
            hps_dict[exp_id] = ast.literal_eval(hps_literal)
    return hps_dict

def hps_to_exp_id(hps, hps_dict):
    matching_exp_id = None
    for exp_id in hps_dict:
        match = True
        for key in hps_dict[exp_id]:
            if hps[key] != hps_dict[exp_id][key]:
                match = False
        if match:
            matching_exp_id = exp_id
            break

    assert matching_exp_id is not None, f"Should have found it! hyp: {hps}"
    return matching_exp_id

def add_gexf_metadata(gexf_file_path, description="", creator="", keywords=""):
    with open(gexf_file_path, 'r') as inp:
        xml_data = inp.readlines()

    data_aug = []
    saw_meta_tag_rep = False
    for line in xml_data:
        line = line.strip()
        if not saw_meta_tag_rep:
            data_aug.append(line)
            if "<meta " in line:
                saw_meta_tag_rep = True     
        else:
            assert "<creator>" in line, f'If this is not true we are not at the part of the file we think we are. We are at line: {line}'
            # we don't inlcude the above, but overwrite it with the below

            data_aug.append(f'  <creator>{creator}</creator>')
            data_aug.append(f'  <description>{description}</description>')
            data_aug.append(f'  <keywords>{keywords}</keywords>')
            saw_meta_tag_rep = False
    
    with open(gexf_file_path, 'w') as out:
        for line in data_aug:
            print(line, file=out)

def do_twm(
        dataset_name,
        kge_model_name,
        run_id,
        hyp_selected,
        model_save_path,
        model_config_path,
        hyps_dict
    ):
    # load hyperparaneters
    if type(hyp_selected) != int:
        exp_id_wanted = hps_to_exp_id(hyp_selected, hyps_dict)
    else:
        exp_id_wanted = hyp_selected
        hyp_selected = hyps_dict[exp_id_wanted]

    # load twig model and data
    TWIG_model = torch.load(model_save_path)
    model_config = load_config(model_config_path)

    twig_data = load_twig_fmt_data(
        dataset_name,
        kge_model_name,
        run_id,
        normalisation=model_config['normalisation'],
        n_bins=model_config['n_bins']
    )
    img_path_pred = f"static/images/TWM-{dataset_name}-{kge_model_name}-{run_id}-pred.svg" #do png, pdf, svg (among maybe others) accepted
    img_path_true = f"static/images/TWM-{dataset_name}-{kge_model_name}-{run_id}-true.svg" #do png, pdf, svg (among maybe others) accepted
    gexf_file_path = f"static/graphs/TWM-{dataset_name}-{kge_model_name}-{run_id}.gexf"
    graph_save_url = f"http://127.0.0.1:5000/static/graphs/" + gexf_file_path

    TWIG_model = torch.load(model_save_path).to(device)
    TWIG_model.eval()
    twm, mrr_pred, mrr_true = create_TWM_graph(
        dataset_name=dataset_name,
        run_id=run_id,
        twig_data=twig_data,
        exp_id_wanted=exp_id_wanted,
        TWIG_model=TWIG_model,
        rescale_y=True,
        graph_split='valid'
    )
    
    # write output
    description = f"dataset :: {dataset_name}\n"
    for i, key in enumerate(hyp_selected):
        val = hyp_selected[key]
        val = str(val).replace('NegativeSampler', '').replace('Loss', '')
        if i == len(hyp_selected) - 1:
            description += f"{key} :: {val}"
        else:
            description += f"{key} :: {val}\n"
    draw_TWM_graph(twm, img_path_pred, draw_pred=True)
    draw_TWM_graph(twm, img_path_true, draw_pred=False)
    nx.write_gexf(twm, gexf_file_path)
    add_gexf_metadata(
        gexf_file_path,
        description=description,
        creator="Jeffrey Seathrún Sardina using TWIG and NetworkX",
        keywords="TWIG, TWM"
    )

    return img_path_pred, img_path_true, graph_save_url, mrr_pred, mrr_true, exp_id_wanted
