# twig-twm imports
from load_data import twm_load
from utils import get_triples, load_custom_dataset, calc_graph_stats, invert_dict

# external imports
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pykeen import datasets
import torch
import ast
import sys
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_twig_fmt_data(dataset_name, norm_func_path):
    twig_data = twm_load(
        dataset_names=[dataset_name],
        normalisation='zscore',
        rescale_y=True,
        dataset_to_run_ids={dataset_name: ['2.1']},
        norm_func_path=norm_func_path
    )
    # os.chdir('../TWM/')
    return twig_data

def create_TWM_graph(
        dataset_name,
        twig_data,
        exp_id_wanted,
        rescale_y,
        graph_split,
        TWIG_model=None,
    ):
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
    for curr_exp_id, batch in enumerate(twig_data[dataset_name]): #trainig and testing are the same hee by manual design
        if curr_exp_id == exp_id_wanted:
            '''
            Note that if X[0,:] is the subject corruption rank being predicted,
            then X[0,:] is the other one being corrupted
            '''
            X, y = batch
            X = X.to(device)
            y = y.to(device)

            '''
            If a TWIG model is not given, use ground truth values
            '''
            if TWIG_model:
                print('Using the given TWIG model')
                R_preds, _ = TWIG_model(X)
            else:
                print('No TWIG model given -- using ground truth values')
                R_preds = y

            max_possible_rank = float(X[0][0])
            if rescale_y:
                if TWIG_model is not None:
                    mrr_pred = torch.mean(1 / (1 + R_preds * (max_possible_rank - 1)))
                else:
                    mrr_pred = torch.mean(1 / (R_preds * max_possible_rank)) # mrr from ranks on range [0,max]
            else:
                mrr_pred = torch.mean(1 / R_preds)
            break
        else:
            pass

    '''
    R preds has the form
        s corr rank
        o corr rank
        s corr rank
        o corr rank
        ....

    We need to turn this into triple learnability. That is --   
        avg((1 / s corr rank), (1 / o corr rank))
        or the average of sided learnabilities
    '''
    assert len(R_preds) % 2 == 0, "should be a multiple of 2 for s and o corruption ranks"

    learnabilities = []
    method = 'inv'
    print(f'Using method: {method}. Note: "inv" is the method outlined in the TWM paper, and should be used to generate all images')
    for i in range(0, len(R_preds), 2):
        if method == 'inv':
            if TWIG_model:
                R_pred_s = (float(R_preds[i]) + 1 / max_possible_rank) * max_possible_rank
                R_pred_o = (float(R_preds[i+1]) + 1 / max_possible_rank) * max_possible_rank
            else:
                R_pred_s = (float(R_preds[i])) * max_possible_rank
                R_pred_o = (float(R_preds[i+1])) * max_possible_rank
            s_learnability = 1 / R_pred_s
            o_learnability = 1 / R_pred_o
            learnability = (s_learnability + o_learnability) / 2
            assert learnability == learnability, 'Learnability should not be NaN!'
            learnability = round(float(learnability), 2)
            print(
                learnability,
                round(float(R_pred_s), 2),
                round(float(R_pred_o), 2),
                round(float(s_learnability), 2),
                round(float(o_learnability), 2),
                sep="\t"
            )
            learnabilities.append(learnability)
        else:
            assert False, f"invalid method: {method}"
    assert len(learnabilities) * 2 == len(R_preds), f"new version should have exactly half the size -- at the  triple level not subj and obj level. But itr was {len(learnabilities)}, len{len(R_preds)}"
        
    '''
    Note: I need to verify triple order is maintained in all of this!
    It *probably* is bubt I am not 100% sure.
    '''
    graph_stats = calc_graph_stats(triples_dicts, do_print=False)
    twm = nx.MultiDiGraph()
    # min_rank = min(R_preds) #also max learnability
    # max_rank = max(R_preds) #also min learnability
    for triple_id, triple in enumerate(triples_dicts[graph_split]):
        s, p, o = triple
        learnability = learnabilities[triple_id]

        # get edge data
        # R_pred = float(R_preds[triple_id])
        # rank_norm = (R_pred - min_rank) / (max_rank - min_rank)
        # if min_rank == max_rank:
        #     learnability = 0
        # else:
        #     learnability = 1 - rank_norm
        # assert learnability == learnability, 'Learnability should not be NaN!'
        # learnability = round(float(learnability), 2)

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
            learnability_score=learnability,
            p_train_freq=pred_freq,
            pyKEEN_ID=p
        )
    return twm, mrr_pred

def draw_TWM_graph(twm, out_file_name):
    edges, learnabilities = zip(
        *nx.get_edge_attributes(twm, 'learnability_score').items()
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
        hyp_selected,
        hyps_dict,
        mode='base',
        use_TWIG=True,
    ):
    model_save_path = "../TWIG/checkpoints/3_v3_CoDExSmall-UMLS-DBpedia50-OpenEA-Kinships_e5-e10.pt"
    norm_func_path = "../TWIG/checkpoints/chkpt-ID_7246262863429676_v3_CoDExSmall-UMLS-DBpedia50-OpenEA-Kinships.normfunc.pkl"
    # MUST BE THE NORM FUNC THAT MATCHES THAT MODEL!!

    if type(hyp_selected) != int:
        exp_id_wanted = hps_to_exp_id(hyp_selected, hyps_dict)
    else:
        exp_id_wanted = hyp_selected
        hyp_selected = hyps_dict[exp_id_wanted]

    description = f"dataset :: {dataset_name}\n"
    for i, key in enumerate(hyp_selected):
        val = hyp_selected[key]
        val = str(val).replace('NegativeSampler', '').replace('Loss', '')
        if i == len(hyp_selected) - 1:
            description += f"{key} :: {val}"
        else:
            description += f"{key} :: {val}\n"

    twig_data = load_twig_fmt_data(dataset_name, norm_func_path)
    out_file_path = f"static/images/TWM-{dataset_name}.svg" #do png, pdf, svg (among maybe others) accepted
    out_file_name = f"TWM-{dataset_name}.svg" #do png, pdf, svg (among maybe others) accepted
    graph_save_url = f"http://127.0.0.1:5000/static/graphs/TWM_{dataset_name}_hyp-{exp_id_wanted}.gexf"
    gexf_file_path = f"static/graphs/TWM_{dataset_name}_hyp-{exp_id_wanted}.gexf"

    if use_TWIG:
        TWIG_model = torch.load(model_save_path).to(device)
        TWIG_model.eval()

    twm, mrr_pred = create_TWM_graph(
        dataset_name,
        twig_data,
        exp_id_wanted,
        rescale_y=True,
        graph_split='valid'
        TWIG_model=TWIG_model if use_TWIG else None,
    )
    print(f'Predicted MRR for the given model configuration is: {mrr_pred}')
    
    draw_TWM_graph(twm, out_file_path)
    nx.write_gexf(twm, gexf_file_path)
    add_gexf_metadata(
        gexf_file_path,
        description=description,
        creator="Jeffrey Seathrún Sardina using TWIG and NetworkX",
        keywords="TWIG, TWM"
    )

    if mode == 'base':
        return out_file_name, graph_save_url, mrr_pred, exp_id_wanted
    else:
        assert False, f"invalid mode: {mode}"

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    hyp_id = int(sys.argv[2])
    use_TWIG = sys.argv[3] == '1'
    hyps_dict = load_hyps_dict("hyp.grid")
    do_twm(
        dataset_name,
        hyp_id,
        hyps_dict,
        use_TWIG=use_TWIG,
    )
