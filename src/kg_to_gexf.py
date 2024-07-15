import networkx as nx
from utils import get_triples, load_custom_dataset, calc_graph_stats
from pykeen import datasets
import sys

def invert_dict(dict_kv):
    dict_vk = {}
    for key in dict_kv:
        val = dict_kv[key]
        assert not val in dict_vk, f'This function assumes a structly 1:1 mapping, but that is not true for {key, val} in the original'
        dict_vk[val] = key
    return dict_vk

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

def create_graph(dataset_name, graph_split):
    try:
        dataset = datasets.get_dataset(dataset=dataset_name)
    except:
        dataset = load_custom_dataset(dataset_name)
    triples_dicts = get_triples(dataset)
    graph_stats = calc_graph_stats(triples_dicts, do_print=False)
    id_to_ent = invert_dict(dataset.entity_to_id)
    id_to_rel = invert_dict(dataset.relation_to_id)

    return graph_from_triples(
        triples_dicts[graph_split],
        graph_stats[graph_split],
        id_to_ent,
        id_to_rel
    )
    
def graph_from_triples(triples, metadata, id_to_ent, id_to_rel):
    graph = nx.MultiDiGraph()
    for s, p, o in triples:
        pred_freq = metadata['pred_freqs'][p]

        # get node data
        s_deg = metadata['degrees'][s]
        s_deg = round(float(s_deg), 2)
        o_deg = metadata['degrees'][o]
        o_deg = round(float(o_deg), 2)

        # annotate graph with that data
        s_node = id_to_ent[s]
        o_node = id_to_ent[o]
        p_rel = id_to_rel[p]
        graph.add_node(s_node, name=s_node, pyKEEN_ID=s, train_deg=s_deg)
        graph.add_node(o_node, name=o_node, pyKEEN_ID=o, train_deg=o_deg)
        graph.add_edge(
            s_node,
            o_node,
            label=p_rel,
            name=p_rel,
            p_train_freq=pred_freq,
            pyKEEN_ID=p
        )
    return graph

def expand(triples_dict, to_expand):
    print(f'{to_expand}')
    expanded_triples = set()
    involved_entities = set()
    for s, p, o in triples_dict:
        if s in to_expand or o in to_expand:
            expanded_triples.add((s,p,o))
            involved_entities.add(s)
            involved_entities.add(o)
    return expanded_triples, involved_entities

def seed_graph(dataset_name, graph_split, node_id, hop_dist):
    try:
        dataset = datasets.get_dataset(dataset=dataset_name)
    except:
        dataset = load_custom_dataset(dataset_name)
    triples_dicts = get_triples(dataset)
    graph_stats = calc_graph_stats(triples_dicts, do_print=False)
    id_to_ent = invert_dict(dataset.entity_to_id)
    id_to_rel = invert_dict(dataset.relation_to_id)

    expanded_triples = set()
    involved_entities = {node_id}
    for _ in range(hop_dist):
        triples, ents = expand(triples_dicts[graph_split], involved_entities)
        expanded_triples |= triples
        involved_entities |= ents

    graph = graph_from_triples(triples, graph_stats[graph_split], id_to_ent, id_to_rel)
    return graph

def main(dataset_name, graph_split, out_path):
    graph = create_graph(dataset_name, graph_split)

    '''
    Median degrees *in the training set only*
    FB15k-237  -- 22.0
    WN18RR     -- 3.0 (no change needed)
    CoDExSmall -- 17.0
    DBpedia50  -- 1.0
    Kinships   -- 164.5
    OpenEA     -- 3.0
    UMLS       -- 58.0
    '''
    # graph = seed_graph(dataset_name, graph_split, 1281, 2) # pykeen id 1281 = fb15k237 /m/01ccr8, a median degree node
    # graph = seed_graph(dataset_name, graph_split, 39075, 3) # pykeen id 39075 = wordnet 14371913, a median degree node
    
    # graph = seed_graph(dataset_name, graph_split, 1007, 2) # pykeen id 1007 = codex Q295502, a median degree node
    # graph = seed_graph(dataset_name, graph_split, 24025, 6) # pykeen id 24025 = dbpedia William_of_Luxi, a median degree node
    # graph = seed_graph(dataset_name, graph_split, 91, 2) # pykeen id 91 = kinships person88, a median degree node
    # graph = seed_graph(dataset_name, graph_split, 3917, 3) # pykeen id 3917 = openea http://dbpedia.org/resource/E258675, a median degree node
    # graph = seed_graph(dataset_name, graph_split, 6, 2) # pykeen id 6 = umls amphibian, a median degree node
    
    nx.write_gexf(graph, out_path)
    add_gexf_metadata(
        out_path,
        description=f'The {dataset_name} KG {graph_split} set :))',
        creator="Jeffrey Seathrún Sardina using PyKEEN and NetworkX",
        keywords="Knowledge Graph"
    )

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    graph_split = sys.argv[2]
    out_path = sys.argv[3]
    main(dataset_name, graph_split, out_path)
