# replication/solvers.py
import numpy as np
import pandas as pd
from scipy.optimize import linprog

try:
    import cvxpy as cp

    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


def apply_round_lot_with_residual_allocation(
        df: pd.DataFrame,
        adjusted_notional: float,
        avail_qty_col: str
) -> pd.DataFrame:
    """
    带整手残差资金再平衡的股数分配：
    1. 基础向下取整到 lot_size。
    2. 计算向下截断释放的资金总额，在富余券源的股票中按优化权重从大到小依次补齐 1 手，压低资金拖累。
    """
    df = df.copy()
    lot_size = df['lot_size'] if 'lot_size' in df.columns else 100.0

    raw_qty = (df['opt_weight'] * adjusted_notional / df['price'])
    base_qty = np.floor(raw_qty / lot_size) * lot_size
    base_qty = np.minimum(base_qty, df[avail_qty_col].fillna(0.0))
    df['opt_qty'] = base_qty
    df['opt_mv'] = df['opt_qty'] * df['price']

    residual_cash = adjusted_notional - df['opt_mv'].sum()
    if residual_cash > 0:
        candidates = df.sort_values(by='opt_weight', ascending=False).index
        for idx in candidates:
            p = df.loc[idx, 'price']
            l_size = df.loc[idx, 'lot_size'] if 'lot_size' in df.columns else 100.0
            one_lot_val = p * l_size

            avail_q = df.loc[idx, avail_qty_col]
            curr_q = df.loc[idx, 'opt_qty']
            if residual_cash >= one_lot_val and (curr_q + l_size) <= avail_q:
                df.loc[idx, 'opt_qty'] += l_size
                df.loc[idx, 'opt_mv'] += one_lot_val
                residual_cash -= one_lot_val

    return df


def _solve_qp(
        N: int,
        w_bmk: np.ndarray,
        w_upper: np.ndarray,
        target_weight: float,
        size_vec: np.ndarray,
        bmk_size_exp: float,
        delta_size: float,
        ind_dummies: np.ndarray,
        bmk_ind_weights: np.ndarray,
        delta_ind: float
):
    """底层 QP (二次规划) 求解引擎"""
    if not HAS_CVXPY:
        print("⚠️ 未安装 cvxpy，跳过 QP 求解。")
        return None

    w = cp.Variable(N)
    objective = cp.Minimize(cp.sum_squares(w - w_bmk))

    constraints = [
        w >= 0.0,
        w <= w_upper,
        cp.sum(w) <= target_weight,
        cp.sum(w) >= target_weight * 0.95,
        size_vec @ w - bmk_size_exp <= delta_size,
        size_vec @ w - bmk_size_exp >= -delta_size,
        cp.norm1(ind_dummies.T @ w - bmk_ind_weights) <= delta_ind
    ]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5)
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return np.maximum(w.value, 0.0)
    except Exception as e:
        print(f"QP 求解异常: {e}")
    return None


def _solve_lp_highs(
        N: int,
        K: int,
        w_bmk: np.ndarray,
        w_upper: np.ndarray,
        target_weight: float,
        size_vec: np.ndarray,
        bmk_size_exp: float,
        delta_size: float,
        ind_dummies: np.ndarray,
        bmk_ind_weights: np.ndarray,
        delta_ind: float,
        enable_elastic: bool,
        lambda_ind: float,
        lambda_size: float
):
    """底层 LP (线性规划 - HiGHS) 求解引擎 (Phase 1 硬约束 + Phase 2 弹性软约束)"""
    idx_w = 0
    idx_u = N
    idx_v = 2 * N
    num_vars_hard = 2 * N + K

    # =========================================================================
    c_hard = np.zeros(num_vars_hard)
    c_hard[idx_u: idx_u + N] = 1.0  # 个股跟踪误差权重
    c_hard[idx_v: idx_v + K] = lambda_ind * 0.5  # 压制行业偏离
    # =========================================================================

    bounds_hard = [(0.0, w_upper[i]) for i in range(N)] + \
                  [(0.0, None) for _ in range(N)] + \
                  [(0.0, None) for _ in range(K)]

    A_ub_hard = []
    b_ub_hard = []

    # 1. 个股偏差
    for i in range(N):
        r1 = np.zeros(num_vars_hard);
        r1[idx_w + i] = 1.0;
        r1[idx_u + i] = -1.0;
        A_ub_hard.append(r1);
        b_ub_hard.append(w_bmk[i])
        r2 = np.zeros(num_vars_hard);
        r2[idx_w + i] = -1.0;
        r2[idx_u + i] = -1.0;
        A_ub_hard.append(r2);
        b_ub_hard.append(-w_bmk[i])

    # 2. 行业偏差
    for k in range(K):
        ind_col = ind_dummies[:, k]
        r1 = np.zeros(num_vars_hard);
        r1[idx_w: idx_w + N] = ind_col;
        r1[idx_v + k] = -1.0;
        A_ub_hard.append(r1);
        b_ub_hard.append(bmk_ind_weights[k])
        r2 = np.zeros(num_vars_hard);
        r2[idx_w: idx_w + N] = -ind_col;
        r2[idx_v + k] = -1.0;
        A_ub_hard.append(r2);
        b_ub_hard.append(-bmk_ind_weights[k])

    r_ind = np.zeros(num_vars_hard);
    r_ind[idx_v: idx_v + K] = 1.0;
    A_ub_hard.append(r_ind);
    b_ub_hard.append(delta_ind)

    # 3. 市值偏离
    r_sz1 = np.zeros(num_vars_hard);
    r_sz1[idx_w: idx_w + N] = size_vec - (bmk_size_exp + delta_size);
    A_ub_hard.append(r_sz1);
    b_ub_hard.append(0.0)
    r_sz2 = np.zeros(num_vars_hard);
    r_sz2[idx_w: idx_w + N] = (bmk_size_exp - delta_size) - size_vec;
    A_ub_hard.append(r_sz2);
    b_ub_hard.append(0.0)

    # 4. 总权重
    r_w1 = np.zeros(num_vars_hard);
    r_w1[idx_w: idx_w + N] = 1.0;
    A_ub_hard.append(r_w1);
    b_ub_hard.append(target_weight)
    r_w2 = np.zeros(num_vars_hard);
    r_w2[idx_w: idx_w + N] = -1.0;
    A_ub_hard.append(r_w2);
    b_ub_hard.append(-(target_weight * 0.95))

    res = linprog(c_hard, A_ub=np.array(A_ub_hard), b_ub=np.array(b_ub_hard), bounds=bounds_hard, method='highs')

    # Phase 2 软约束回退
    if not res.success and enable_elastic:
        print("⚠️ LP 硬约束无可行解，进入 Phase 2 弹性软约束罚项求解...")
        idx_sp = 2 * N + K
        idx_sn = 2 * N + K + 1
        num_vars_soft = 2 * N + K + 2

        c_soft = np.zeros(num_vars_soft)
        c_soft[idx_u: idx_u + N] = 1.0
        c_soft[idx_v: idx_v + K] = lambda_ind
        c_soft[idx_sp] = lambda_size
        c_soft[idx_sn] = lambda_size

        bounds_soft = [(0.0, w_upper[i]) for i in range(N)] + \
                      [(0.0, None) for _ in range(N)] + \
                      [(0.0, None) for _ in range(K)] + \
                      [(0.0, None), (0.0, None)]

        A_ub_soft = []
        b_ub_soft = []

        for i in range(N):
            r1 = np.zeros(num_vars_soft);
            r1[idx_w + i] = 1.0;
            r1[idx_u + i] = -1.0;
            A_ub_soft.append(r1);
            b_ub_soft.append(w_bmk[i])
            r2 = np.zeros(num_vars_soft);
            r2[idx_w + i] = -1.0;
            r2[idx_u + i] = -1.0;
            A_ub_soft.append(r2);
            b_ub_soft.append(-w_bmk[i])

        for k in range(K):
            ind_col = ind_dummies[:, k]
            r1 = np.zeros(num_vars_soft);
            r1[idx_w: idx_w + N] = ind_col;
            r1[idx_v + k] = -1.0;
            A_ub_soft.append(r1);
            b_ub_soft.append(bmk_ind_weights[k])
            r2 = np.zeros(num_vars_soft);
            r2[idx_w: idx_w + N] = -ind_col;
            r2[idx_v + k] = -1.0;
            A_ub_soft.append(r2);
            b_ub_soft.append(-bmk_ind_weights[k])

        A_eq_soft = []
        b_eq_soft = []
        r_sz_eq = np.zeros(num_vars_soft)
        r_sz_eq[idx_w: idx_w + N] = size_vec - bmk_size_exp
        r_sz_eq[idx_sp] = -1.0
        r_sz_eq[idx_sn] = 1.0
        A_eq_soft.append(r_sz_eq);
        b_eq_soft.append(0.0)

        r_tot = np.zeros(num_vars_soft);
        r_tot[idx_w: idx_w + N] = 1.0;
        A_ub_soft.append(r_tot);
        b_ub_soft.append(target_weight)
        r_tot_low = np.zeros(num_vars_soft);
        r_tot_low[idx_w: idx_w + N] = -1.0;
        A_ub_soft.append(r_tot_low);
        b_ub_soft.append(-(target_weight * 0.90))

        res = linprog(c_soft, A_ub=np.array(A_ub_soft), b_ub=np.array(b_ub_soft), A_eq=np.array(A_eq_soft),
                      b_eq=np.array(b_eq_soft), bounds=bounds_soft, method='highs')

        if not res.success:
            print("⚠️ 进一步松弛总仓位下限，确保输出最优解...")
            A_ub_soft.pop()
            b_ub_soft.pop()
            res = linprog(c_soft, A_ub=np.array(A_ub_soft), b_ub=np.array(b_ub_soft), A_eq=np.array(A_eq_soft),
                          b_eq=np.array(b_eq_soft), bounds=bounds_soft, method='highs')

    if not res.success:
        raise RuntimeError(f"❌ LP 优化求解失败: {res.message}")

    return res.x[idx_w: idx_w + N]

def solve_hedging_portfolio(
        df_model: pd.DataFrame,
        adjusted_notional: float,
        solver_type: str = 'LP',
        delta_size: float = 0.02,
        delta_ind: float = 0.05,
        overweight_factor: float = 1.2,
        enable_elastic: bool = True,
        lambda_ind: float = 8.0,
        lambda_size: float = 30.0
):
    """
    通用对冲组合优化入口 (对外标准统一接口)
    """
    df = df_model.copy().reset_index(drop=True)
    N = len(df)

    # 1. 因子与特征计算
    ln_cap = np.log(np.maximum(df['market_cap'].values, 1.0))
    std_cap = ln_cap.std()
    df['size_zscore'] = (ln_cap - ln_cap.mean()) / (std_cap if std_cap > 0 else 1.0)

    avail_qty_col = 'effective_avail_qty' if 'effective_avail_qty' in df.columns else 'internal_avail_qty'
    df['effective_avail_mv'] = df[avail_qty_col] * df['price']
    df['w_upper'] = np.minimum(df['weight'] * overweight_factor, df['effective_avail_mv'] / adjusted_notional)
    df['w_upper'] = np.maximum(df['w_upper'].fillna(0.0), 0.0)

    industries = sorted(df['bics_level_2'].dropna().unique())
    K = len(industries)
    ind_dummies = (
        pd.get_dummies(df['bics_level_2'], dtype=float)
        .reindex(columns=industries, fill_value=0.0)
        .values.astype(float)
    )

    w_bmk = df['weight'].values.astype(float)
    bmk_ind_weights = ind_dummies.T @ w_bmk
    bmk_size_exp = float(np.dot(w_bmk, df['size_zscore'].values))
    size_vec = df['size_zscore'].values.astype(float)

    total_upper = float(df['w_upper'].sum())
    target_weight = min(1.0, total_upper)

    # 2. 分发求解引擎
    opt_w = None
    solver_mode = solver_type.upper()

    if solver_mode == 'QP':
        print(f"⚙️ 尝试调用 QP (二次规划) 引擎...")
        opt_w = _solve_qp(
            N=N, w_bmk=w_bmk, w_upper=df['w_upper'].values,
            target_weight=target_weight, size_vec=size_vec,
            bmk_size_exp=bmk_size_exp, delta_size=delta_size,
            ind_dummies=ind_dummies, bmk_ind_weights=bmk_ind_weights,
            delta_ind=delta_ind
        )
        if opt_w is None:
            print("⚠️ QP 无可行解或未配置，自动无缝降级回退至 LP 引擎...")
            solver_mode = 'LP'

    if solver_mode == 'LP':
        print(f"⚙️ 调用 LP (HiGHS) 引擎...")
        opt_w = _solve_lp_highs(
            N=N, K=K, w_bmk=w_bmk, w_upper=df['w_upper'].values,
            target_weight=target_weight, size_vec=size_vec,
            bmk_size_exp=bmk_size_exp, delta_size=delta_size,
            ind_dummies=ind_dummies, bmk_ind_weights=bmk_ind_weights,
            delta_ind=delta_ind, enable_elastic=enable_elastic,
            lambda_ind=lambda_ind, lambda_size=lambda_size
        )

    # 3. 规整与风控回算
    df['opt_weight'] = np.round(opt_w, 6)
    df = apply_round_lot_with_residual_allocation(df, adjusted_notional, avail_qty_col)

    tot_weight = df['opt_weight'].sum()
    opt_size_exp = float(
        np.dot(df['opt_weight'].values / tot_weight, df['size_zscore'].values)) if tot_weight > 0 else 0.0
    opt_ind_weights = ind_dummies.T @ df['opt_weight'].values
    actual_ind_dev = float(np.sum(np.abs(opt_ind_weights - bmk_ind_weights)))

    return df, opt_size_exp, bmk_size_exp, actual_ind_dev