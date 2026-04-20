from sklearn.cluster import KMeans
import numpy as np
import torch
from tqdm import tqdm


def get_flatten_features(model, data_loader, args):
    progress_bar_eval = tqdm(range(len(data_loader)))
    flatten_hidden_state_history = []

    with torch.inference_mode():
        for batch in data_loader:
            batch = {
                'input_ids': batch['input_ids'].cuda(),
                'labels': batch['labels'].cuda(),
                'attention_mask': batch['attention_mask'].cuda()
            }

            outputs = model(**batch, output_hidden_states=True)

            tmp_hidden_state = []

            for idx_layer in range(len(outputs.hidden_states)):
                hidden_state_cur_layer = outputs.hidden_states[idx_layer][0]

                if args.feature_token == 'avg':
                    vec = torch.mean(hidden_state_cur_layer, dim=0)
                else:
                    vec = hidden_state_cur_layer[-1]

                tmp_hidden_state.append(vec.cpu().numpy())

            flatten_hidden_state_history.append(
                np.array(tmp_hidden_state, dtype=np.float64)
                .reshape(-1)
            )

            progress_bar_eval.update(1)
            progress_bar_eval.set_description("extracting feature")

    return flatten_hidden_state_history


def clustering(features, args):
    return _cluster(features, args) + (features,)


def _cluster(features, args):
    if args.clustering.lower() == 'kmeans':
        kmeans = KMeans(
            n_clusters=args.n_cluster,
            max_iter=2000,
            n_init=10,
            init='k-means++'
        )
        cluster_labels = kmeans.fit_predict(features)
        centroids = kmeans.cluster_centers_
        return cluster_labels, centroids

    elif args.clustering.lower() == 'hdbscan':
        from hdbscan import HDBSCAN

        clusterer = HDBSCAN(
            min_cluster_size=args.min_cluster,
            allow_single_cluster=False
        )
        cluster_labels = clusterer.fit_predict(features)

        centroids = []
        for cid in np.unique(cluster_labels):
            if cid == -1:
                continue
            centroids.append(features[cluster_labels == cid].mean(axis=0))

        return cluster_labels, np.array(centroids)

    else:
        raise ValueError(f"Unknown clustering method: {args.clustering}")
