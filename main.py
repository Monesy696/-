import sys

from data_processing import (
    export_q1,
    export_q2,
    export_q3,
    export_q4,
    load_boundary,
    load_radius_history,
)
from plot_results import (
    plot_boundary,
    plot_q1_profiles,
    plot_q2_profiles,
    plot_q3_endpoint,
    plot_q4_analysis,
)
from validation import (
    run_q1_validation,
    run_q2_validation,
    run_q3_validation,
    run_q4_validation,
)


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

    q3_solution, q3_convergence, q3_comparison, q3_time_check, q3_sensitivity, q3_summary = (
        run_q3_validation(boundary)
    )
    result3_path, table5 = export_q3(q3_solution)
    plot_q3_endpoint(q3_solution, q3_sensitivity)

    radius_history = load_radius_history()
    (
        q4_solution,
        q4_convergence,
        q4_comparison,
        q4_time_check,
        q4_sensitivity,
        q4_decomposition,
        q4_summary,
    ) = run_q4_validation(boundary, radius_history, q3_solution=q3_solution)
    result4_path, table6 = export_q4(q4_solution)
    plot_q4_analysis(
        q4_solution,
        radius_history,
        q4_decomposition,
        q4_sensitivity,
    )

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
    print("Q3求解完成：附录3变物性FVM–BDF，表面加密640区间")
    print(f"连续终点：{q3_solution['endpoint_s'] / 3600.0:.8f} h")
    print(f"首个严格达标60 s时刻：{q3_solution['strict_output_s'] / 3600.0:.8f} h")
    print(f"result3文件：{result3_path}")
    print("Q3终点网格收敛：")
    print(q3_convergence.to_string(index=False))
    print("Q3正文表5：")
    print(table5.round(4).to_string(index=False))
    print("Q3图表：outputs/figures/figure_04_q3_endpoint.*、figure_05_q3_sensitivity.*")
    print("Q4求解完成：附录4材料坐标收缩域FVM–BDF，表面加密320区间")
    print(f"连续终点：{q4_solution['endpoint_s'] / 3600.0:.8f} h")
    print(f"首个严格达标60 s时刻：{q4_solution['strict_output_s'] / 3600.0:.8f} h")
    print(f"终点半径：{q4_solution['endpoint_radius_m'] * 100.0:.6f} cm")
    print(f"result4文件：{result4_path}")
    print("Q4终点网格收敛：")
    print(q4_convergence.to_string(index=False))
    print("Q4物性与几何作用分解：")
    print(q4_decomposition.to_string(index=False))
    print("Q4正文表6：")
    print(table6.round(4).to_string(index=False))
    print("Q4图表：outputs/figures/figure_06_q4_effects.*、figure_07_q4_profiles.*")


if __name__ == "__main__":
    main()
