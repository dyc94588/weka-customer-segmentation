#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成与项目同构的演示数据集。

原始数据集 Online_Retail.csv 为课程提供的模拟零售交易数据（500 条客户记录、
10 个行为特征）。由于原数据不在本仓库分发，这里用固定随机种子复现一份
**结构完全一致**的演示数据，方便任何人 clone 后直接跑通 WEKA 的完整流程。

数据字典（10 个特征，5 数值 + 5 类别）：
  TotalAmount      数值  累计消费金额（元）
  OrderCount       数值  订单数
  ActiveDays       数值  活跃天数
  ProductVariety   数值  购买商品品类数
  AvgOrderValue    数值  客单价 = TotalAmount / OrderCount
  Country          类别  客户所在国家
  Channel          类别  主要获客渠道
  PaymentMethod    类别  主要支付方式
  ProductCategory  类别  主要购买品类
  CustomerValue    类别  客户价值标签（High / Medium / Low），作为分类目标

标签并非随机生成，而是由 AvgOrderValue 与 ActiveDays 的加权得分决定
（score = 0.6 × 归一化客单价 + 0.4 × 归一化活跃天数，按分位数切三档），
对应项目中 J48 决策树学到的业务规则（根节点 AvgOrderValue > 0.568）。

用法：
    python scripts/generate_data.py            # 输出 data/online_retail_demo.csv
"""
import csv
import os
import random

SEED = 20231212
N = 500

COUNTRIES = ["United Kingdom", "Germany", "France", "Australia", "USA", "Netherlands"]
CHANNELS = ["Organic Search", "Paid Ads", "Referral", "Social Media", "Email"]
PAYMENTS = ["Credit Card", "PayPal", "Bank Transfer", "Debit Card"]
CATEGORIES = ["Home Decor", "Kitchenware", "Giftware", "Stationery", "Textiles"]


def main() -> None:
    rng = random.Random(SEED)
    records = []

    for cid in range(1, N + 1):
        # 潜在活跃度与消费力：两条相互独立的基础分布
        activity = rng.betavariate(2.2, 2.0)      # 活跃度 0~1
        spending = rng.betavariate(2.0, 2.4)      # 消费力 0~1

        active_days = max(1, int(round(activity * 118 + rng.gauss(0, 4))))
        active_days = min(active_days, 120)

        order_count = max(1, int(round(activity * 14 + rng.gauss(0, 1.2))))
        avg_order_value = round(120 + spending * 1680 + rng.gauss(0, 70), 2)
        avg_order_value = max(35.0, avg_order_value)

        total_amount = round(avg_order_value * order_count, 2)
        product_variety = max(1, min(5, int(round(spending * 4 + activity * 1.5 + rng.gauss(0, 0.7)))))

        records.append({
            "CustomerID": f"C{cid:04d}",
            "TotalAmount": total_amount,
            "OrderCount": order_count,
            "ActiveDays": active_days,
            "ProductVariety": product_variety,
            "AvgOrderValue": avg_order_value,
            "Country": rng.choice(COUNTRIES),
            "Channel": rng.choice(CHANNELS),
            "PaymentMethod": rng.choices(PAYMENTS, weights=[5, 2, 1, 3])[0],
            "ProductCategory": rng.choice(CATEGORIES),
        })

    # 按「归一化客单价 × 活跃天数」的加权得分切三档，保证三类大致均衡
    aovs = [r["AvgOrderValue"] for r in records]
    days = [r["ActiveDays"] for r in records]
    aov_min, aov_max = min(aovs), max(aovs)
    day_min, day_max = min(days), max(days)

    def score(r):
        norm_aov = (r["AvgOrderValue"] - aov_min) / (aov_max - aov_min)
        norm_day = (r["ActiveDays"] - day_min) / (day_max - day_min)
        return 0.6 * norm_aov + 0.4 * norm_day

    for r in records:
        r["_score"] = score(r)

    ranked = sorted(r["_score"] for r in records)
    low_cut = ranked[int(len(ranked) * 0.30)]
    high_cut = ranked[int(len(ranked) * 0.68)]

    for r in records:
        if r["_score"] >= high_cut:
            r["CustomerValue"] = "High"
        elif r["_score"] < low_cut:
            r["CustomerValue"] = "Low"
        else:
            r["CustomerValue"] = "Medium"
        r.pop("_score")

    rows = []
    for r in records:
        rows.append({
            "CustomerID": r["CustomerID"],
            "TotalAmount": f'{r["TotalAmount"]:.2f}',
            "OrderCount": r["OrderCount"],
            "ActiveDays": r["ActiveDays"],
            "ProductVariety": r["ProductVariety"],
            "AvgOrderValue": f'{r["AvgOrderValue"]:.2f}',
            "Country": r["Country"],
            "Channel": r["Channel"],
            "PaymentMethod": r["PaymentMethod"],
            "ProductCategory": r["ProductCategory"],
            "CustomerValue": r["CustomerValue"],
        })

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "online_retail_demo.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    dist = {}
    for r in rows:
        dist[r["CustomerValue"]] = dist.get(r["CustomerValue"], 0) + 1

    print(f"已生成 {len(rows)} 条记录 -> {out_path}")
    print("客户价值分布：", dist)


if __name__ == "__main__":
    main()
