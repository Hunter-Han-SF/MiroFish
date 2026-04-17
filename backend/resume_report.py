"""断点续传报告生成脚本"""
from app import create_app
from app.services.report_agent import ReportAgent, ReportManager
from app.utils.locale import set_locale

set_locale('zh')
app = create_app()

with app.app_context():
    r = ReportManager.get_report('report_caae56665df0')
    agent = ReportAgent(
        graph_id=r.graph_id,
        simulation_id=r.simulation_id,
        simulation_requirement=r.simulation_requirement,
    )
    result = agent.generate_report(report_id='report_caae56665df0')
    print(f'Status: {result.status.value}')
    if result.error:
        print(f'Error: {result.error}')
    else:
        print('SUCCESS!')
