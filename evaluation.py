import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config
from progress.bar import Bar
from scipy import stats
from sklearn import metrics
from pickle import load
from model.nns import *
from model.bls import BroadLearningSystem
from model.elm import ELMRegressor, GenELMRegressor
from model.random_layer import (
    RandomLayer,
    MLPRandomLayer,
    RBFRandomLayer,
    GRBFRandomLayer,
)
from model.pyESN import ESN
import argparse


def get_gamma_zeta_mu(y_pred, y_true):
    # gamma
    se = np.square(y_pred - y_true)
    mse = np.mean(se, axis=2)
    rse = np.sqrt(mse)
    # zeta
    e0 = np.linalg.norm((y_true), axis=(2))
    e1 = np.linalg.norm((y_pred - y_true), axis=(2))
    norm_error = np.divide(e1, e0)
    # mu
    d0 = np.linalg.norm(y_pred, ord=1, axis=(2))
    d1 = np.linalg.norm(y_true, ord=1, axis=(2))
    d1 = d0 + d1
    d2 = np.linalg.norm(y_pred - y_true, ord=1, axis=(2))
    sape = np.divide(d2, d1)
    return rse, norm_error, sape


def get_effective_range(metric, threshold, name=""):
    avg_metric = np.mean(metric, axis=0)
    # print(f"max {max(sample_avg)}; min {min(sample_avg)}")
    avg_effective_range = np.argmax(avg_metric > threshold, axis=0)
    logtic_array = metric > threshold
    logtic_array[:, -1] = True
    effective_range = np.argmax(logtic_array, axis=0)
    #     effective_range_mean  = np.mean(effective_range)
    # print (f"{name} max effective range is  {np.max(effective_range)}; top 5 ER at {effective_range.argsort()[:5][::-1]}")
    return avg_effective_range, avg_metric


def metric_calculation(
    y_pred,
    y_true,
    gamma=0.4,
    mu=0.8,
    zeta=0.4,
    norm=False,
    dataset_name="",
    model_name="",
):  # alpha=0.4, beta = 0.8,gamma=0.4,rho = 11.4
    name = f"{model_name} on {dataset_name}:"
    print(f"{name}, y_true shape {y_true.shape}; y_pred shape {y_pred.shape}")
    n_samples, n_vars = y_true.shape[0], y_true.shape[-1]
    nan_idx = np.isnan(y_true)
    y_true = y_true[~nan_idx].reshape((n_samples, -1, n_vars))
    y_pred = y_pred[~nan_idx].reshape((n_samples, -1, n_vars))
    if norm:
        scaler = load(
            open(f"{config.DATA_FOLDER}/{dataset_name}/stanard_scaler.pkl", "rb")
        )
        y_pred_shape = y_pred.shape
        y_pred_reshaped = y_pred.reshape((-1, n_vars))
        y_pred = scaler.inverse_transform(y_pred_reshaped).reshape(y_pred_shape)

    y_pred_reshaped = y_pred.reshape((-1, 1))
    y_true_reshaped = y_true.reshape((-1, 1))

    rse, norm_error, sape = get_gamma_zeta_mu(y_pred, y_true)
    rmse = np.mean(rse)
    range_gamma, avg_rse = get_effective_range(rse, gamma, "rmse of " + name)
    # norm error nu
    norm_error_mean = np.mean(norm_error)
    range_zeta, avg_ne = get_effective_range(norm_error, zeta, "ne of " + name)
    # smape
    smape = np.mean(sape)
    range_mu, avg_sape = get_effective_range(sape, mu, "smape of " + name)
    # 1-step lyapunov loss lambda
    # d1 = np.linalg.norm((y_pred[:,:-1]-y_true[:,:-1]), axis=(2))
    # d2 = np.linalg.norm((y_pred[:,1:]-y_true[:,1:]), axis=(2))
    # lyapunov_error = np.divide(d2,d1)
    # lyapunov_error_mean = np.mean(lyapunov_error)
    # Normalized Joint Prediction Error(NJPE)
    # njpe = 0
    # diff = np.subtract((y_pred - y_true))
    # np.amax(y_true, axis=0)
    # e1 = np.linalg.norm((y_pred - y_true), axis=(2))
    # for i in range(n_vars):
    #     pred = y_pred[:,:,i].reshape((-1,1))
    #     true = y_true[:,:,i].reshape((-1,1))
    #     max_y, min_y = np.max(true), np.min(true)
    #     njpe += np.mean(((true - pred)/(max_y - min_y))**2)
    # njpe = njpe/n_vars

    # print (f"MAE: {mae}")
    # print (f"RMSE: {rmse}")
    # print (f"$\gamma$: {range_gamma}")
    # print (f"SMAPE: {smape}")
    # print (f"$\mu$: {range_mu}")
    # print (f"NJPE: {njpe}")
    # print (f"Mean norm error: {norm_error_mean}")
    # print (f"$\zeta$: {range_zeta}")
    # print (f"Mean 1-step lyapunov loss: {lyapunov_error_mean}")
    # print (f"max effective range is  {np.max(effective_range)}; top 20 effective range at {effective_range.argsort()[-20:][::-1]}")
    return (
        [rmse, range_gamma, avg_rse],
        [norm_error_mean, range_zeta, avg_ne],
        [smape, range_mu, avg_sape],
        y_true,
        y_pred,
    )  # mae, lyapunov_error_mean,  njpe,


def evaluation(
    y_pred,
    y_true,
    performance=None,
    gamma=0.4,
    mu=0.8,
    zeta=0.4,
    norm=False,
    dataset_name="",
    model_name="",
):
    rmse, ne, smape, y_true, y_pred = metric_calculation(
        y_pred, y_true, gamma, mu, zeta, norm, dataset_name, model_name
    )

    if performance is None:
        performance_idx = [
            "$\mathbb{E}_\gamma$",
            "$\mathbb{P}_\gamma$",
            "$\mathbb{E}_\zeta$",
            "$\mathbb{P}_\zeta$",
            "$\mathbb{E}_\mu$",
            "$\mathbb{P}_\mu$",
        ]
        performance = pd.DataFrame([], index=performance_idx)
    performance[model_name] = [rmse[0], rmse[1], ne[0], ne[1], smape[0], smape[1]]
    return performance, rmse, ne, smape, y_true, y_pred


def get_pred_bls_elm(saved_folder, train_samples, test_inputs, pred_steps, x_y_lag):
    if os.path.exists(
        f"{config.RESULT_FOLDER}/{saved_folder}/bls.npy"
    ) and os.path.exists(f"{config.RESULT_FOLDER}/{saved_folder}/elm.npy"):
        pred_bls = np.load(f"{config.RESULT_FOLDER}/{saved_folder}/bls.npy")
        pred_elm = np.load(f"{config.RESULT_FOLDER}/{saved_folder}/elm.npy")
    else:
        assert (
            len(train_samples.shape) == 3
        ), "train samples should be 3 dimensions: (samples, time steps, feature dimensions)"
        assert (
            len(test_inputs.shape) == 3
        ), "test input sequence should be 3 dimensions: (samples, time steps, feature dimensions)"
        x_steps = test_inputs.shape[1]
        ns_tr, nv_tr = train_samples.shape[0], train_samples.shape[-1]
        ns_te, n_steps, nv_te = test_inputs.shape
        pred_shape = (ns_te, x_steps + pred_steps, nv_tr)
        pred_bls = np.zeros(pred_shape)
        pred_elm = np.zeros(pred_shape)
        bls = BroadLearningSystem(saved_folder=f"{config.MODEL_FOLDER}/{saved_folder}")
        rhl = RandomLayer(n_hidden=500, activation_func="tanh", alpha=1)
        elm = GenELMRegressor(hidden_layer=rhl)
        # for i in range(nv_tr):
        train_x = train_samples[:, :x_steps].reshape((ns_tr, -1))
        train_y = train_samples[:, x_y_lag : x_steps + x_y_lag].reshape((ns_tr, -1))
        # train_x = train_samples_i[:,:x_steps]#.reshape((ns_tr,-1))
        # train_y = train_samples_i[:,x_y_lag:x_steps+x_y_lag]#.reshape((ns_tr,-1))
        test_x = test_inputs.reshape((ns_te, -1))
        bls.training(train_x, train_y)
        prediction = bls.inference(test_x, pred_steps * nv_te, x_y_lag * nv_te)
        pred_bls = prediction.reshape(pred_shape)
        elm.fit(train_x, train_y)
        pred_elm = elm.inference(test_x, pred_steps * nv_te, x_y_lag * nv_te).reshape(
            pred_shape
        )

        np.save(f"{config.RESULT_FOLDER}/{saved_folder}/bls", pred_bls)
        np.save(f"{config.RESULT_FOLDER}/{saved_folder}/elm", pred_elm)
    return pred_bls, pred_elm


def get_pred_esn(saved_folder, train_samples, test_inputs, pred_steps, x_y_lag):
    if os.path.exists(f"{config.RESULT_FOLDER}/{saved_folder}/esn.npy"):
        pred_esn = np.load(f"{config.RESULT_FOLDER}/{saved_folder}/esn.npy")
    else:
        n_sample, nv_tr = train_samples.shape[0], train_samples.shape[-1]
        esn = ESN(
            n_inputs=nv_tr,
            n_outputs=nv_tr,
            n_reservoir=500,
            sparsity=0.2,
            random_state=23,
            spectral_radius=1.2,
            noise=0.0005,
        )
        x_steps = test_inputs.shape[1]
        # y_pred = np.zeros((n_sample,gen_len+x_steps,n_var))
        j = 0
        for sample in Bar("generating").iter(train_samples):
            if j % 100 == 0:
                print(f"training... {100.0*j/n_sample}% completed")
            train_x = sample[:x_steps]
            train_y = sample[x_y_lag : x_steps + x_y_lag]
            esn.fit(train_x, train_y)
            j += 1
        pred_esn = np.zeros((test_inputs.shape[0], x_steps + pred_steps, nv_tr))
        j = 0
        for test_input in Bar("generating").iter(test_inputs):
            if j % 100 == 0:
                print(f"generating ESN prediction... {100.0*j/n_sample}% completed")
            pred_esn[j] = esn.inference(test_input, pred_steps, x_y_lag)
        np.save(f"{config.RESULT_FOLDER}/{saved_folder}/esn", pred_esn)
    return pred_esn


def get_pred_mt(saved_folder, test_inputs, pred_steps, x_y_lag):
    if os.path.exists(f"{config.RESULT_FOLDER}/{saved_folder}/mt.npy"):
        pred_mt = np.load(f"{config.RESULT_FOLDER}/{saved_folder}/mt.npy")
    else:
        mtran = MusicTransformer(
            test_inputs.shape[-1],
            f"{config.MODEL_FOLDER}/{saved_folder}",
            x_y_lag=x_y_lag,
        )
        pred_mt = mtran.inference(test_inputs, pred_steps=pred_steps)
        if pred_mt is not None:
            np.save(f"{config.RESULT_FOLDER}/{saved_folder}/mt", pred_mt)
    return pred_mt


def get_pred_tf(saved_folder, test_inputs, pred_steps, x_y_lag):
    if os.path.exists(f"{config.RESULT_FOLDER}/{saved_folder}/tf.npy"):
        pred_tf = np.load(f"{config.RESULT_FOLDER}/{saved_folder}/tf.npy")
    else:
        tf = TeacherForcing(
            test_inputs.shape[-1],
            f"{config.MODEL_FOLDER}/{saved_folder}",
            x_y_lag=x_y_lag,
        )
        pred_tf = tf.inference(test_inputs, pred_steps=pred_steps)
        if pred_tf is not None:
            np.save(f"{config.RESULT_FOLDER}/{saved_folder}/tf", pred_tf)
    return pred_tf


def get_pred_naive(saved_folder, test_inputs, pred_steps, x_y_lag, k=20):
    if os.path.exists(f"{config.RESULT_FOLDER}/{saved_folder}/hf{k}naive.npy"):
        pred_hf = np.load(f"{config.RESULT_FOLDER}/{saved_folder}/hf{k}naive.npy")
    else:
        hf = HorizonForcing(
            test_inputs.shape[-1],
            f"{config.MODEL_FOLDER}/{saved_folder}",
            x_y_lag=x_y_lag,
            k=k,
            suffix="naive",
        )
        print(hf)
        pred_hf = hf.inference(test_inputs, pred_steps=pred_steps)
        if pred_hf is not None:
            np.save(f"{config.RESULT_FOLDER}/{saved_folder}/hf{k}naive", pred_hf)
    return pred_hf


# def get_pred_ss(saved_folder, test_inputs, pred_steps, x_y_lag):
#     if os.path.exists(f"{config.RESULT_FOLDER}/{saved_folder}/ss.npy"):
#         pred_ss = np.load(f"{config.RESULT_FOLDER}/{saved_folder}/ss.npy")
#     else:
#         ss = ScheduledSampling(test_inputs.shape[-1], f"{config.MODEL_FOLDER}/{saved_folder}",x_y_lag=x_y_lag,ts = test_inputs.shape[1])
#         pred_ss = ss.inference(test_inputs, pred_steps=pred_steps)
#         if pred_ss is not None:
#             np.save(f"{config.RESULT_FOLDER}/{saved_folder}/ss",pred_ss)
#     return pred_ss
def get_pred_dss(saved_folder, test_inputs, pred_steps, x_y_lag, decay="is"):
    if os.path.exists(f"{config.RESULT_FOLDER}/{saved_folder}/dss_{decay}.npy"):
        pred_ss = np.load(f"{config.RESULT_FOLDER}/{saved_folder}/dss_{decay}.npy")
    else:
        ss = DiscontinuousScheduledSampling(
            test_inputs.shape[-1],
            f"{config.MODEL_FOLDER}/{saved_folder}",
            x_y_lag=x_y_lag,
            ts=test_inputs.shape[1],
            decay=decay,
        )
        pred_ss = ss.inference(test_inputs, pred_steps=pred_steps)
        if pred_ss is not None:
            np.save(f"{config.RESULT_FOLDER}/{saved_folder}/dss_{decay}", pred_ss)
    return pred_ss


def get_pred_hf(saved_folder, test_inputs, pred_steps, x_y_lag, k=5):
    if os.path.exists(f"{config.RESULT_FOLDER}/{saved_folder}/hf{k}.npy"):
        pred_hf = np.load(f"{config.RESULT_FOLDER}/{saved_folder}/hf{k}.npy")
    else:
        hf = HorizonForcing(
            test_inputs.shape[-1],
            f"{config.MODEL_FOLDER}/{saved_folder}",
            x_y_lag=x_y_lag,
            k=k,
        )
        print(hf)
        pred_hf = hf.inference(test_inputs, pred_steps=pred_steps)
        if pred_hf is not None:
            np.save(f"{config.RESULT_FOLDER}/{saved_folder}/hf{k}", pred_hf)
    return pred_hf


def get_pred_true_informer(sys, inference_steps):
    result_folder = f"{config.RESULT_FOLDER}/informer/{sys}_ftM_sl100_ll10_pl{inference_steps}_dm512_nh8_el3_dl2_df2048_atprob_fc3_ebtimeF_dtTrue_mxTrue_test_0/"
    y_true = np.load(f"{result_folder}true.npy")
    pred = np.load(f"{result_folder}pred.npy")
    return pred, y_true


def get_pred_true_autoformer(sys, inference_steps):
    result_folder = f"{config.RESULT_FOLDER}/autoformer/{sys}_ftM_sl100_ll10_pl{inference_steps}_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_test_0/"
    y_true = np.load(f"{result_folder}true.npy")
    pred = np.load(f"{result_folder}pred.npy")
    return pred, y_true


ETT_ABLATION = {
    "Teacher Forcing": {"fun": get_pred_tf},
    "ETT Direct": {"fun": get_pred_naive, "factor": 20},
    "HF20": {"fun": get_pred_hf, "factor": 20},
}
HF_STUDY = {
    "HF5": {"fun": get_pred_hf, "factor": 5},
    "HF10": {"fun": get_pred_hf, "factor": 10},
    "HF15": {"fun": get_pred_hf, "factor": 15},
    "HF20": {"fun": get_pred_hf, "factor": 20},
}
RECURRENT = {
    "BLS": {"fun": get_pred_bls_elm},
    "ELM": {"fun": get_pred_bls_elm},
    #   "SSES":{"fun":get_pred_ss},
    "SSFS": {"fun": get_pred_dss, "factor": "is"},
    "HF5": {"fun": get_pred_hf, "factor": 5},
    "HF10": {"fun": get_pred_hf, "factor": 10},
    "HF15": {"fun": get_pred_hf, "factor": 15},
    "HF20": {"fun": get_pred_hf, "factor": 20},
}
TRANSFORMERS = {
    "MTF": {"fun": get_pred_mt},
    "IF": {"fun": get_pred_true_informer},
    "AF": {"fun": get_pred_true_autoformer},
    "HF5": {"fun": get_pred_hf, "factor": 5},
    "HF10": {"fun": get_pred_hf, "factor": 10},
    "HF15": {"fun": get_pred_hf, "factor": 15},
    "HF20": {"fun": get_pred_hf, "factor": 20},
}
BENCHMARKING = {
    "BLS": {"fun": get_pred_bls_elm},
    "ELM": {"fun": get_pred_bls_elm},
    #   "SSES":{"fun":get_pred_ss},
    "SSFS": {"fun": get_pred_dss, "factor": "is"},
    "MTF": {"fun": get_pred_mt},
    "IF": {"fun": get_pred_true_informer},
    "AF": {"fun": get_pred_true_autoformer},
    "HF5": {"fun": get_pred_hf, "factor": 5},
    "HF10": {"fun": get_pred_hf, "factor": 10},
    "HF15": {"fun": get_pred_hf, "factor": 15},
    "HF20": {"fun": get_pred_hf, "factor": 20},
}
ALL = {
    "BLS": {"fun": get_pred_bls_elm},
    "ELM": {"fun": get_pred_bls_elm},
    #   "SSES":{"fun":get_pred_ss},
    "SSFS": {"fun": get_pred_dss, "factor": "is"},
    "MTF": {"fun": get_pred_mt},
    "IF": {"fun": get_pred_true_informer},
    "AF": {"fun": get_pred_true_autoformer},
    "HF5": {"fun": get_pred_hf, "factor": 5},
    "HF10": {"fun": get_pred_hf, "factor": 10},
    "HF15": {"fun": get_pred_hf, "factor": 15},
    "HF20": {"fun": get_pred_hf, "factor": 20},
}
NO_AUTO_INFORMER = {
    "BLS": {"fun": get_pred_bls_elm},
    "ELM": {"fun": get_pred_bls_elm},
    #   "SSES":{"fun":get_pred_ss},
    "SSFS": {"fun": get_pred_dss, "factor": "is"},
    "MTF": {"fun": get_pred_mt},
    # "IF": {"fun": get_pred_true_informer},
    # "AF": {"fun": get_pred_true_autoformer},
    "HF5": {"fun": get_pred_hf, "factor": 5},
    "HF10": {"fun": get_pred_hf, "factor": 10},
    # "HF15": {"fun": get_pred_hf, "factor": 15},
    # "HF20": {"fun": get_pred_hf, "factor": 20},
}
MODE = {
    "ett": ETT_ABLATION,
    "hf": HF_STUDY,
    "benchmark": BENCHMARKING,
    "all": ALL,
    "recurrent": RECURRENT,
    "transformers": TRANSFORMERS,
    "no-auto-informer": NO_AUTO_INFORMER,
}


def _plot_effective_range(
    dataframe,
    ax,
    palette,
    y_lim,
    x_lim,
    y_label="Error $\gamma$",
    delta=None,
    anchor=None,
    fontsize=24,
):
    g = sns.lineplot(
        x="index",
        y="value",
        hue="variable",
        palette=palette,
        errorbar="sd",
        linewidth=5,
        data=pd.melt(dataframe, "index"),
        ax=ax,
    )
    g.set(ylim=y_lim)
    g.set(xlim=(0, x_lim))
    if delta:
        g.axhline(delta[0], ls="--")
    g.set_ylabel(y_label, fontsize=fontsize)
    g.tick_params(axis="both", which="major", labelsize=fontsize)
    # place the legend outside the figure/plot
    if anchor:
        g.legend(bbox_to_anchor=anchor)
    else:
        ax.get_legend().remove()


def plot_comparison(
    study_mode,
    figsize,
    fontsize=24,
    linewidth=3,
    anchor=(0, -0.05, 1, 1),
    legsize=2,
    fold=1,
    x_y_lag=1,
):
    normalization = False
    result = {}
    pred_funs = MODE[study_mode]
    model_names = pred_funs.keys()
    num_models, num_dataset = len(pred_funs), len(config.EXP_SETTING)
    fig, axs = plt.subplots(nrows=3, ncols=num_dataset, figsize=figsize, sharex="col")

    def get_results(
        pred,
        y_true,
        perform,
        rmse_out,
        ne_out,
        smape_out,
        idx,
        dataset_name,
        model_name,
        norm,
    ):
        print(
            f"here 2 {dataset_name}, {model_name} pred shape is {pred.shape}; y_true shape is {y_true.shape}"
        )
        perform, rmse, ne, smape, y_true, pred_bls = evaluation(
            pred, y_true, perform, gamma, mu, zeta, norm, dataset_name, model_name
        )
        rmse_out[:, idx], ne_out[:, idx], smape_out[:, idx] = rmse[2], ne[2], smape[2]
        return perform, rmse_out, ne_out, smape_out

    for idx_data, (sys, setting) in enumerate(config.EXP_SETTING.items()):
        for sub, params in setting.items():
            key, x_steps, inference_steps = (
                params["title"],
                params["input_steps"],
                params["inference_steps"],
            )
            gamma, zeta, mu, y_lim = (
                params["gamma"],
                params["zeta"],
                params["mu"],
                params["y_lim"][study_mode],
            )

            dataset_name = config.get_dataset_name(sys, sub)
            train_samples, test_inputs, y_true = load_model_config(
                dataset_name, fold, normalization, x_steps
            )
            print(
                f"train_samples {train_samples.shape}, test_inputs {test_inputs.shape}, y_true {y_true.shape}"
            )
            y_true = y_true[:, :inference_steps]
            saved_folder = config.get_model_name(
                dataset_name, fold, normalization, x_y_lag
            )
            if not os.path.exists(f"{config.RESULT_FOLDER}/{saved_folder}"):
                os.makedirs(f"{config.RESULT_FOLDER}/{saved_folder}")
            perform = None
            rmse_out, ne_out, smape_out = None, None, None
            colors = []
            pred_bls, pred_elm = None, None
            for idx_pred, (name, vals) in enumerate(pred_funs.items()):
                if rmse_out is None:
                    rmse_out, ne_out, smape_out = (
                        np.empty((inference_steps, num_models)),
                        np.empty((inference_steps, num_models)),
                        np.empty((inference_steps, num_models)),
                    )
                    rmse_out[:], ne_out[:], smape_out[:] = np.nan, np.nan, np.nan
                if name == "BLS":
                    if pred_bls is None:
                        pred_bls, pred_elm = vals["fun"](
                            saved_folder,
                            train_samples,
                            test_inputs,
                            inference_steps,
                            x_y_lag,
                        )
                    perform, rmse_out, ne_out, smape_out = get_results(
                        pred_bls[:, x_steps : x_steps + inference_steps],
                        y_true,
                        perform,
                        rmse_out,
                        ne_out,
                        smape_out,
                        idx_pred,
                        dataset_name,
                        "BLS",
                        normalization,
                    )
                    colors.append(config.COLOR_BANK["BLS"])
                elif name == "ELM":
                    if pred_elm is None:
                        pred_bls, pred_elm = vals["fun"](
                            saved_folder,
                            train_samples,
                            test_inputs,
                            inference_steps,
                            x_y_lag,
                        )
                    perform, rmse_out, ne_out, smape_out = get_results(
                        pred_elm[:, x_steps : x_steps + inference_steps],
                        y_true,
                        perform,
                        rmse_out,
                        ne_out,
                        smape_out,
                        idx_pred,
                        dataset_name,
                        "ELM",
                        normalization,
                    )
                    colors.append(config.COLOR_BANK["ELM"])
                elif name == "IF":
                    pred, true = vals["fun"](sys, inference_steps)
                    perform, rmse_out, ne_out, smape_out = get_results(
                        pred,
                        true,
                        perform,
                        rmse_out,
                        ne_out,
                        smape_out,
                        idx_pred,
                        dataset_name,
                        "Informer",
                        normalization,
                    )
                    colors.append(config.COLOR_BANK["IF"])
                elif name == "AF":
                    pred, true = vals["fun"](sys, inference_steps)
                    perform, rmse_out, ne_out, smape_out = get_results(
                        pred,
                        true,
                        perform,
                        rmse_out,
                        ne_out,
                        smape_out,
                        idx_pred,
                        dataset_name,
                        "Autoformer",
                        normalization,
                    )
                    colors.append(config.COLOR_BANK["AF"])

                else:
                    if vals.get("factor") is not None:
                        pred = vals["fun"](
                            saved_folder,
                            test_inputs,
                            inference_steps,
                            x_y_lag,
                            vals["factor"],
                        )
                    else:
                        pred = vals["fun"](
                            saved_folder, test_inputs, inference_steps, x_y_lag
                        )
                    if pred is not None:
                        print(
                            f"{dataset_name}, {name}, pred shape is {pred.shape}; y_true shape is {y_true.shape}; inference_steps {inference_steps}"
                        )
                        perform, rmse_out, ne_out, smape_out = get_results(
                            pred[:, x_steps : x_steps + inference_steps],
                            y_true,
                            perform,
                            rmse_out,
                            ne_out,
                            smape_out,
                            idx_pred,
                            dataset_name,
                            name,
                            normalization,
                        )
                    colors.append(config.COLOR_BANK[name])
            rmse_out_df = pd.DataFrame(rmse_out, columns=model_names).reset_index()
            ne_out_df = pd.DataFrame(ne_out, columns=model_names).reset_index()
            smape_out_df = pd.DataFrame(smape_out, columns=model_names).reset_index()
            palette = sns.color_palette(colors)
            _plot_effective_range(
                rmse_out_df,
                axs[0, idx_data],
                palette,
                y_lim[0],
                inference_steps,
                y_label="$Error$ $\gamma$",
                fontsize=fontsize,
            )
            _plot_effective_range(
                ne_out_df,
                axs[1, idx_data],
                palette,
                y_lim[1],
                inference_steps,
                y_label="$Error$ $\zeta$",
                fontsize=fontsize,
            )
            _plot_effective_range(
                smape_out_df,
                axs[2, idx_data],
                palette,
                y_lim[2],
                inference_steps,
                y_label="$Error$ $\mu$",
                fontsize=fontsize,
            )
            # if idx_data!=0:
            axs[0, idx_data].yaxis.label.set_visible(False)
            axs[1, idx_data].yaxis.label.set_visible(False)
            axs[2, idx_data].yaxis.label.set_visible(False)
            # axs[0,idx_data].set_title(key)
            axs[2, idx_data].xaxis.label.set_visible(False)

            result[key] = perform

    df_result = pd.concat(result)
    df_result = df_result.round(3)
    handles, labels = axs[-1, -1].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc=("lower center"),
        bbox_to_anchor=anchor,
        bbox_transform=plt.gcf().transFigure,
        ncol=num_models // 2 + 1,
        labelspacing=0.1,
        fontsize=fontsize + 5,
        frameon=False,
    )
    for legobj in leg.legendHandles:
        legobj.set_linewidth(legsize)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.1)

    return df_result


def load_model_config(dataset_name, fold, norm=False, x_steps=100):
    if config.PRETRAINED:
        dataset_name = f"pre_generated/{dataset_name}"
    test_samples = np.load(f"{config.DATA_FOLDER}/{dataset_name}/test.npy")
    train_samples = np.load(f"{config.DATA_FOLDER}/{dataset_name}/train.npy")
    path = os.path.join(
        f"{config.DATA_FOLDER}/{dataset_name}/indices",
        f"{config.FOLDS}fold.{fold}_train_index.npy",
    )
    train_index = np.load(path)
    train_samples = train_samples[train_index]
    if len(test_samples.shape) == 2:
        test_samples = np.expand_dims(test_samples, axis=0)
    test_input = test_samples[:, :x_steps]
    y_true = test_samples[:, x_steps:]
    if norm:
        train_shape = train_samples.shape
        test_shape = test_samples.shape
        n_variables = train_shape[-1]
        train_reshaped = train_samples.reshape((-1, n_variables))
        test_reshaped = test_samples.reshape((-1, n_variables))
        scaler = load(
            open(f"{config.DATA_FOLDER}/{dataset_name}/stanard_scaler.pkl", "rb")
        )
        train_samples = scaler.transform(train_reshaped).reshape(train_shape)
        test_samples = scaler.transform(test_reshaped).reshape(test_shape)
        test_input = test_samples[:, :x_steps]
    return train_samples, test_input, y_true
