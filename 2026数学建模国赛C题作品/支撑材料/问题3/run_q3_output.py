# -*- coding: utf-8 -*-
"""
运行MPC并输出result3.xlsx（按官方模板格式）
===================================================
修复项：
  ① 路径改为本机附件目录 + 官方模板(附件5/result3.xlsx)
  ② 时段单元格单位修复：LP变量为功率(kW)，结果表要求kWh，写表时统一×DT
  ③ 计划购电量表的"全天购电费"改为按计划量计算（price×plan×DT），
     调整购电量表的"全天购电费"保留最终总费用（含调整费+紧急购电）
  ④ 增加求解器状态检查：非Optimal时打印告警，避免异常数据静默写成0
"""
import os, numpy as np, pandas as pd, pulp, time, warnings
from datetime import datetime, timedelta
import openpyxl
warnings.filterwarnings('ignore')

# ============ 路径配置（基于脚本所在目录）============
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(CURRENT_DIR, "..", "附件"))     # 附件1/2/3所在目录（Andy\附件）
TEMPLATE = os.path.join(BASE, '附件5', 'result3.xlsx')              # 官方模板
OUT = os.path.join(CURRENT_DIR, 'result3.xlsx')                     # 输出到本脚本同目录
# ================================================

# 参数（附录1）
CAP_MAX = 10800.0; CAP_MIN = 1200.0; P_MAX = 5000.0; ETA = 0.9
SOC_INIT = 6000.0; DT = 10.0/60.0; N = 144
EM = 5.0; DEC_PEN = 0.5; INC_EXT = 0.5

# 读数据
price = pd.read_excel(os.path.join(BASE,'附件1.xlsx'))['电价'].values.astype(float)
df_l = pd.read_excel(os.path.join(BASE,'附件2.xlsx'),sheet_name='小区负载')
df_p = pd.read_excel(os.path.join(BASE,'附件2.xlsx'),sheet_name='光伏发电实际功率')
dates_l = pd.to_datetime(df_l.iloc[:,0]).values
d2i = {pd.Timestamp(d).strftime('%Y-%m-%d'):i for i,d in enumerate(dates_l)}
load_data = df_l.iloc[:,1:145].values.astype(float)
pv_data = df_p.iloc[:,1:145].values.astype(float)

df3 = pd.read_excel(os.path.join(BASE,'附件3.xlsx'))
df3['日期'] = df3['日期'].ffill()
fc = {}
for _,row in df3.iterrows():
    try:
        k = pd.to_datetime(str(row['日期']).strip()).strftime('%Y-%m-%d')
    except: continue
    h = int(str(row['预报时刻']).strip().split(':')[0])
    fc[(k,h)] = row.iloc[2:26].values.astype(float)

def get_fc(key, fch, cur=None):
    if (key,fch) not in fc: return np.zeros(N)
    hourly = fc[(key,fch)]
    full = np.zeros(25)
    for h in range(24):
        t = fch+1+h
        if 0<=t<=24: full[t]=hourly[h]
    if cur is not None: full[fch]=cur
    res = np.zeros(N)
    for i in range(N):
        tm=(i+1)*10; th=tm/60.0
        h0=int(np.floor(th)); h1=min(h0+1,24)
        res[i]=full[h0]+(th-h0)*(full[h1]-full[h0])
    return res

def solve_lp(price, load, pv, soc0, planned=None, si=0, ei=N, fixed=None, past_pv=None, tag=''):
    n_opt = ei-si
    p = pulp.LpProblem("m", pulp.LpMinimize)
    chg=[pulp.LpVariable(f"c{i}",lowBound=0,upBound=P_MAX) for i in range(N)]
    dis=[pulp.LpVariable(f"d{i}",lowBound=0,upBound=P_MAX) for i in range(N)]
    emg=[pulp.LpVariable(f"e{i}",lowBound=0) for i in range(N)]
    buy=[pulp.LpVariable(f"b{i}",lowBound=0) for i in range(n_opt)]
    soc=[pulp.LpVariable(f"s{i}",lowBound=CAP_MIN,upBound=CAP_MAX) for i in range(N+1)]
    p += soc[0]==soc0
    p += soc[N]==soc0  # 每日SOC闭环
    obj = pulp.lpSum(price[si+i]*buy[i]*DT for i in range(n_opt))
    obj += pulp.lpSum(EM*price[i]*emg[i]*DT for i in range(N))
    if planned is not None:
        wd=[pulp.LpVariable(f"wd{i}",lowBound=0) for i in range(n_opt)]
        wi=[pulp.LpVariable(f"wi{i}",lowBound=0) for i in range(n_opt)]
        for i in range(n_opt):
            p += wd[i] >= planned[si+i]-buy[i]
            p += wi[i] >= buy[i]-planned[si+i]
        obj += pulp.lpSum(DEC_PEN*price[si+i]*wd[i]*DT+INC_EXT*price[si+i]*wi[i]*DT for i in range(n_opt))
    p += obj
    pvu = past_pv if past_pv is not None else pv
    for i in range(N):
        if i<si:
            bv = fixed[i] if fixed is not None else 0
            p += bv+pvu[i]+dis[i]+emg[i] >= load[i]+chg[i]
        else:
            p += buy[i-si]+pv[i]+dis[i]+emg[i] >= load[i]+chg[i]
        p += soc[i+1]==soc[i]+chg[i]*DT*ETA-dis[i]*DT/ETA
    p.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
    if pulp.LpStatus[p.status] != 'Optimal':   # ✅ 状态检查
        print(f"  ⚠️ [{tag}] 求解非最优: {pulp.LpStatus[p.status]}")
    br = np.zeros(N)
    if fixed is not None: br[:si]=fixed[:si]
    for i in range(n_opt): br[si+i]=pulp.value(buy[i])
    return br, np.array([pulp.value(chg[i]) for i in range(N)]), \
           np.array([pulp.value(dis[i]) for i in range(N)]), \
           np.array([pulp.value(emg[i]) for i in range(N)]), \
           pulp.value(soc[N]), pulp.value(p.objective)

def settle(price, load, pva, buy, soc0, tag=''):
    p = pulp.LpProblem("st", pulp.LpMinimize)
    chg=[pulp.LpVariable(f"c{i}",lowBound=0,upBound=P_MAX) for i in range(N)]
    dis=[pulp.LpVariable(f"d{i}",lowBound=0,upBound=P_MAX) for i in range(N)]
    emg=[pulp.LpVariable(f"e{i}",lowBound=0) for i in range(N)]
    soc=[pulp.LpVariable(f"s{i}",lowBound=CAP_MIN,upBound=CAP_MAX) for i in range(N+1)]
    p += soc[0]==soc0; p += soc[N]==soc0
    p += pulp.lpSum(EM*price[i]*emg[i]*DT for i in range(N))
    for i in range(N):
        p += buy[i]+pva[i]+dis[i]+emg[i] >= load[i]+chg[i]
        p += soc[i+1]==soc[i]+chg[i]*DT*ETA-dis[i]*DT/ETA
    p.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
    if pulp.LpStatus[p.status] != 'Optimal':   # ✅ 状态检查
        print(f"  ⚠️ [{tag}] 结算非最优: {pulp.LpStatus[p.status]}")
    return np.array([pulp.value(chg[i]) for i in range(N)]), \
           np.array([pulp.value(dis[i]) for i in range(N)]), \
           np.array([pulp.value(emg[i]) for i in range(N)]), pulp.value(soc[N])

# 1月预热
soc = SOC_INIT
for d in [datetime(2025,1,1)+timedelta(days=i) for i in range(31)]:
    k = d.strftime('%Y-%m-%d')
    if k not in d2i: continue
    idx = d2i[k]
    p,_,_,_,_,_ = solve_lp(price, load_data[idx], get_fc(k,0), soc, tag=f"预热{k}")
    _,_,_,soc = settle(price, load_data[idx], pv_data[idx], p, soc, tag=f"预热{k}")
print(f"2/1 SOC={soc:.0f}")

# 复制官方模板
import shutil
os.makedirs(os.path.dirname(OUT), exist_ok=True)
shutil.copy(TEMPLATE, OUT)
wb = openpyxl.load_workbook(OUT)

dates = [datetime(2025,2,1)+timedelta(days=i) for i in range(334)]
all_plan = []; all_adj = []; all_chg4=[]; all_dis4=[]; all_soc0=[]; all_soc24=[]; all_emg=[]; all_cost=[]; all_plan_cost=[]

t0=time.time()
for di, d in enumerate(dates):
    k = d.strftime('%Y-%m-%d')
    if k not in d2i: continue
    idx = d2i[k]; ld=load_data[idx]; pva=pv_data[idx]

    # 0:00计划
    pb,_,_,_,_,_ = solve_lp(price, ld, get_fc(k,0), soc, tag=f"{k}计划")
    ab = pb.copy()
    for ah in [6,12,18]:
        si2 = ah*6
        fcv = get_fc(k, ah, cur=pva[si2])
        nb,_,_,_,_,_ = solve_lp(price, ld, fcv, soc, planned=pb, si=si2, ei=N, fixed=ab, past_pv=pva, tag=f"{k}@{ah}点")
        nxt = (ah+6)*6 if ah<18 else N
        ab[si2:nxt] = nb[si2:nxt]

    soc0_day = soc
    fc2,fd2,fe2,sf = settle(price, ld, pva, ab, soc, tag=k)
    soc = sf

    # 费用（乘DT）
    bc = np.sum(price*ab*DT)
    dp = DEC_PEN*np.sum(price[ab<pb]*(pb[ab<pb]-ab[ab<pb])*DT)
    ip = INC_EXT*np.sum(price[ab>pb]*(ab[ab>pb]-pb[ab>pb])*DT)
    ec = np.sum(EM*price*fe2*DT)
    all_plan.append(pb); all_adj.append(ab)
    all_soc0.append(soc0_day); all_soc24.append(sf)
    all_cost.append(bc+dp+ip+ec)
    all_plan_cost.append(np.sum(price*pb)*DT)   # 计划购电费

    # 4h充放电（kWh）
    c4=np.zeros(6); d4=np.zeros(6)
    for seg in range(6):
        c4[seg]=np.sum(fc2[seg*24:(seg+1)*24]*DT)
        d4[seg]=np.sum(fd2[seg*24:(seg+1)*24]*DT)
    all_chg4.append(c4); all_dis4.append(d4)

    # 紧急购电段
    i=0
    while i<N:
        if fe2[i]>0.1:
            si3=i
            while i<N and fe2[i]>0.1: i+=1
            tot = np.sum(fe2[si3:i]*DT)
            sm2=si3*10; em3=i*10
            sh2,smv2=divmod(sm2,60); eh2,emv2=divmod(em3,60)
            all_emg.append((d, f"{sh2}:{smv2:02d}-{eh2}:{emv2:02d}", tot))
        else: i+=1

    if (di+1)%50==0:
        print(f"  [{di+1}/334] {k} 累计={sum(all_cost)/1e4:.0f}万 ({time.time()-t0:.0f}s)")

# === 写Sheet1 计划购电量（时段=kWh，费用列=计划购电费）===
ws1 = wb['计划购电量']
for di, d in enumerate(dates):
    r = di+2
    ws1.cell(r,1,d)
    for t in range(N):
        ws1.cell(r,t+2, round(float(all_plan[di][t])*DT,2))   # ✅ kW→kWh
    ws1.cell(r,N+2, round(float(np.sum(all_plan[di])*DT),2))
    ws1.cell(r,N+3, round(float(all_plan_cost[di]),2))        # ✅ 计划购电费

# === Sheet2 调整购电量（时段=kWh，费用列=最终总费用）===
ws2 = wb['调整购电量']
for di, d in enumerate(dates):
    r = di+2
    ws2.cell(r,1,d)
    for t in range(N):
        ws2.cell(r,t+2, round(float(all_adj[di][t])*DT,2))    # ✅ kW→kWh
    ws2.cell(r,N+2, round(float(np.sum(all_adj[di])*DT),2))
    ws2.cell(r,N+3, round(float(all_cost[di]),2))

# === Sheet3 充放电量 ===
ws3 = wb['充放电量']
for row in ws3.iter_rows(min_row=2, max_row=ws3.max_row):
    for c in row: c.value = None
seg_lab = ['0:00-4:00','4:00-8:00','8:00-12:00','12:00-16:00','16:00-20:00','20:00-24:00']
r = 2
for di, d in enumerate(dates):
    for seg in range(6):
        if seg==0:
            ws3.cell(r,1,d); ws3.cell(r,5,datetime(2025,d.month,d.day,0,0).time())
            ws3.cell(r,6,round(float(all_soc0[di]),2))
        ws3.cell(r,2,seg_lab[seg])
        ws3.cell(r,3,round(float(all_chg4[di][seg]),2))
        ws3.cell(r,4,round(float(all_dis4[di][seg]),2))
        if seg==1:
            ws3.cell(r,5,'24:00'); ws3.cell(r,6,round(float(all_soc24[di]),2))
        r+=1

# === Sheet4 紧急购电量 ===
ws4 = wb['紧急购电量']
for row in ws4.iter_rows(min_row=2, max_row=ws4.max_row):
    for c in row: c.value = None
r = 2; cur_date = None
for d, ts, amt in all_emg:
    if d != cur_date:
        ws4.cell(r,1,d); cur_date = d
    ws4.cell(r,2,ts); ws4.cell(r,3,round(float(amt),2))
    r+=1

wb.save(OUT)
print(f"\n已保存: {OUT}")
print(f"全年总成本: {sum(all_cost):.2f}元 = {sum(all_cost)/1e4:.1f}万")
print(f"总用时: {time.time()-t0:.0f}s")
