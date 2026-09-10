from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_and_process_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_and_process_data,
                inputs="raw_waymo_data",
                outputs=dict(
                    clean_data="clean_waymo_data",
                    frontend_data="frontend_waymo_json"
                ),
                name="process_waymo_data_node",
            )
        ]
    )
