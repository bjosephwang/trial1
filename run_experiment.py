import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import argparse
from lib.stackedDAE import StackedDAE
from lib.dec import DEC
from lib.dec_f import DEC_F
from lib.dcn import DeepClusteringNetwork
from lib.utils import acc
from lib.datasets import daily
from sklearn import metrics
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
foo = foo
dataset = "daily"
repeat = 5
batch_size = 256
latent_dim = 3
n_clusters = 3

acc_km = []
nmi_km = []
entropy_km = []
balance_km = []

acc_dcn = []
nmi_dcn = []
entropy_dcn = []
balance_dcn = []

acc_db = []
nmi_db = []
entropy_db = []
balance_db = []

acc_ae_km = []
nmi_ae_km = []
entropy_ae_km = []
balance_ae_km = []

acc_dec = []
nmi_dec = []
entropy_dec = []
balance_dec = []

acc_ours = []
nmi_ours = []
entropy_ours = []
balance_ours = []


def compute_entropy(n_clusters, y_pred, yt, nbins):
    total_entropy = 0
    for k in range(n_clusters):
        idx = np.where(y_pred == k)
        y_k = np.squeeze(yt[idx])
        cluster_size = np.size(idx)
        hist = np.zeros((nbins,))
        for i in range(nbins):
            val = i+1
            hist[i] = np.size(np.where(y_k == val))
        gt_hist = np.ones(nbins) * cluster_size / nbins
        hist = hist/np.sum(hist)
        gt_hist = gt_hist/np.sum(gt_hist)
        entropy_k = entropy(hist, gt_hist)
        total_entropy += entropy_k
    return total_entropy

for i in range(1, repeat+1):
    print("Experiment #%d" % i)

    # train_loader = torch.utils.data.DataLoader(
    #     daily('/data/wbo/daily', train=True),
    #     batch_size=batch_size, shuffle=True, num_workers=0)
    # test_loader = torch.utils.data.DataLoader(
    #     daily('/data/wbo/daily', train=False),
    #     batch_size=batch_size, shuffle=False, num_workers=0)
    # # pretrain
    # sdae = StackedDAE(input_dim=405, z_dim=latent_dim, binary=False,
    #     encodeLayer=[500,500,2000], decodeLayer=[2000,500,500], activation="relu",
    #     dropout=0)
    # sdae.pretrain(train_loader, test_loader, lr=0.1, batch_size=batch_size,
    #     num_epochs=100, corrupt=0.2, loss_type="mse")
    # sdae.fit(train_loader, test_loader, lr=0.1, num_epochs=100, corrupt=0.2, loss_type="mse")
    # #sdae_savepath = ("model/sdae-run-%d.pt" % i)
    sdae_savepath = ("model/sdae-run-1.pt")
    # sdae.save_model(sdae_savepath)

    # finetune
    daily_train = daily('/data/wbo/daily', train=True)
    daily_test = daily('/data/wbo/daily', train=False)
    X = daily_train.train_data
    y = daily_train.train_y
    attr = daily_train.train_attrs
    Xt = daily_test.test_data
    yt = daily_test.test_y.cpu().numpy()
    attr_t = daily_test.test_attrs.cpu().numpy()
    attr_array = np.squeeze(attr.cpu().numpy())
    uniq_vals = np.unique(attr_array)
    n_bins = np.size(uniq_vals)


    # DCN
    dcn = DeepClusteringNetwork(input_dim=405, z_dim=latent_dim, n_centroids=n_clusters, binary=False,
                                 encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu",
                                dropout=0)
    dcn.load_model(sdae_savepath)
    dcn.fit(X, y, lr=0.001, batch_size=batch_size, num_epochs=10)
    # testing
    testdata, _ = dcn.forward(Xt)
    kmeans = KMeans(n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(testdata.detach().cpu().numpy())
    print((metrics.normalized_mutual_info_score(yt, y_pred)))
    print(acc(yt, y_pred))
    total_entropy = compute_entropy(n_clusters, y_pred, attr_t, n_bins)
    print("Total entropy: %.5f" % total_entropy)

    acc_dcn.append(acc(yt, y_pred))
    nmi_dcn.append(metrics.normalized_mutual_info_score(yt, y_pred))
    entropy_dcn.append(total_entropy)


    dec = DEC(input_dim=405, z_dim=latent_dim, n_clusters=n_clusters,
        encodeLayer=[500,500,2000], activation="relu", dropout=0)
    #print(dec)
    # dec.load_model(sdae_savepath)

    # Ae+kmeans

    testdata, _= dec.forward(Xt.cpu())

    # testing AE+kmeans
    kmeans = KMeans(n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(testdata.detach().cpu().numpy())
    print((metrics.normalized_mutual_info_score(yt, y_pred)))
    print(acc(yt, y_pred))
    total_entropy = compute_entropy(n_clusters, y_pred, attr_t, n_bins)
    print("Total entropy: %.5f" % total_entropy)

    acc_ae_km.append(acc(yt, y_pred))
    nmi_ae_km.append(metrics.normalized_mutual_info_score(yt, y_pred))
    entropy_ae_km.append(total_entropy)

    # testing kmeans
    kmeans = KMeans(n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(Xt.cpu().numpy())
    print((metrics.normalized_mutual_info_score(yt, y_pred)))
    print(acc(yt, y_pred))
    total_entropy = compute_entropy(n_clusters, y_pred, attr_t, n_bins)
    print("Total entropy: %.5f" % total_entropy)

    acc_km.append(acc(yt, y_pred))
    nmi_km.append(metrics.normalized_mutual_info_score(yt, y_pred))
    entropy_km.append(total_entropy)


    # DEC
    dec = DEC(input_dim=405, z_dim=latent_dim, n_clusters=n_clusters,
                  encodeLayer=[500, 500, 2000], activation="relu", dropout=0)
    dec.load_model(sdae_savepath)
    dec.fit(X, y, lr=0.01, batch_size=batch_size, num_epochs=30,
        update_interval=1, km=kmeans)
    # dec_savepath = ("model/dec-run-%d.pt" % i)
    # dec.save_model(dec_savepath)
    # testing
    _, q = dec.forward(Xt)
    y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
    print((metrics.normalized_mutual_info_score(yt, y_pred)))
    print(acc(yt, y_pred))

    total_entropy = compute_entropy(n_clusters, y_pred, attr_t, n_bins)
    print("Total entropy: %.5f" % total_entropy)

    acc_dec.append(acc(yt, y_pred))
    nmi_dec.append(metrics.normalized_mutual_info_score(yt, y_pred))
    entropy_dec.append(total_entropy)

    # DEC_F

    dec_f = DEC_F(input_dim=405, z_dim=latent_dim, n_clusters=n_clusters, n_bins=n_bins,
                  encodeLayer=[500, 500, 2000], activation="relu", dropout=0)
    dec_f.load_model(sdae_savepath)

    dec_f.fit(X, y, attr, lr=0.01, batch_size=batch_size, num_epochs=30,
              update_interval=1, km=kmeans)
    # testing
    testdata, q, _ = dec_f.forward(Xt)
    y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
    print((metrics.normalized_mutual_info_score(yt, y_pred)))
    print(acc(yt, y_pred))

    total_entropy = compute_entropy(n_clusters, y_pred, attr_t, n_bins)
    print("Total entropy: %.5f" % total_entropy)

    acc_ours.append(acc(yt, y_pred))
    nmi_ours.append(metrics.normalized_mutual_info_score(yt, y_pred))
    entropy_ours.append(total_entropy)


print("kmeans: ACC: %s + %s; NMI:  %s + %s; Fair: %s + %s;" % (np.mean(np.array(acc_km)), np.std(np.array(acc_km)),
      np.mean(np.array(nmi_km)), np.std(np.array(nmi_km)), np.mean(np.array(entropy_km)), np.std(np.array(entropy_km))))
print("DCN: ACC: %s + %s; NMI:  %s + %s; Fair: %s + %s;" % (np.mean(np.array(acc_dcn)), np.std(np.array(acc_dcn)),
      np.mean(np.array(nmi_dcn)), np.std(np.array(nmi_dcn)), np.mean(np.array(entropy_dcn)), np.std(np.array(entropy_dcn))))
print("AE+kmeans: ACC: %s + %s; NMI:  %s + %s; Fair: %s + %s;" % (np.mean(np.array(acc_ae_km)), np.std(np.array(acc_ae_km)),
      np.mean(np.array(nmi_ae_km)), np.std(np.array(nmi_ae_km)), np.mean(np.array(entropy_ae_km)), np.std(np.array(entropy_ae_km))))
print("DEC: ACC: %s + %s; NMI:  %s + %s; Fair: %s + %s;" % (np.mean(np.array(acc_dec)), np.std(np.array(acc_dec)),
      np.mean(np.array(nmi_dec)), np.std(np.array(nmi_dec)), np.mean(np.array(entropy_dec)), np.std(np.array(entropy_dec))))
print("Ours: ACC: %s + %s; NMI:  %s + %s; Fair: %s + %s;" % (np.mean(np.array(acc_ours)), np.std(np.array(acc_ours)),
      np.mean(np.array(nmi_ours)), np.std(np.array(nmi_ours)), np.mean(np.array(entropy_ours)), np.std(np.array(entropy_ours))))