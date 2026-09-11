import sys

from data_processing import export_q1, export_q2, load_boundary
from plot_results import plot_boundary, plot_q1_profiles, plot_q2_profiles
from validation import run_q1_validation, run_q2_validation


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    boundary = load_boundary()
    q1_solution, convergence, comparison, validation_summary = run_q1_validation(boundary)
    result_path, table_temperature, table_moisture = export_q1(q1_solution)
    plot_boundary(boundary)
    plot_q1_profiles(q1_solution)

    q2_solution, q2_convergence, q2_comparison, q2_summary = run_q2_validation(boundary)
    result2_path, table3, table4 = export_q2(q2_solution)
    plot_q2_profiles(q2_solution)

    print("Q1求解完成：FVM–BDF，正式网格320个径向区间")
    print(f"result1文件：{result_path}")
    print("网格收敛：")
    print(convergence.to_string(index=False))
    print("BDF与独立CN–Picard复核：")
    print(comparison.to_string(index=False))
    print("验证摘要：")
    print(validation_summary.to_string(index=False))
    print("正文表1末行：")
    print(table_temperature.tail(1).round(4).to_string(index=False))
    print("正文表2末行：")
    print(table_moisture.tail(1).round(4).to_string(index=False))
    print("Q1图表：outputs/figures/figure_01_boundary.*、figure_02_q1_profiles.*")
    print("Q2求解完成：变物性FVM–BDF，正式网格320个径向区间")
    print(f"result2文件：{result2_path}")
    print("Q2网格收敛：")
    print(q2_convergence.to_string(index=False))
    print("Q2 BDF与Radau复核：")
    print(q2_comparison.to_string(index=False))
    print("Q2验证摘要：")
    print(q2_summary.to_string(index=False))
    print("正文表3：")
    print(table3.round(4).to_string(index=False))
    print("正文表4：")
    print(table4.round(4).to_string(index=False))
    print("Q2图表：outputs/figures/figure_03_q2_profiles.*")


if __name__ == "__main__":
    main()
