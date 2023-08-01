from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import ParameterFloat


def create_condition_step(evaluate_model_step, evaluation_report, register_model_step):
    accuracy_threshold = ParameterFloat(name="accuracy_threshold", default_value=0.70)

    fail_step = FailStep(
        name="fail",
        error_message=Join(
            on=" ",
            values=[
                "Execution failed because the model's accuracy was lower than",
                accuracy_threshold,
            ],
        ),
    )

    condition_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluate_model_step.name,
            property_file=evaluation_report,
            json_path="metrics.accuracy.value",
        ),
        right=accuracy_threshold,
    )

    condition_step = ConditionStep(
        name="check-model-accuracy",
        conditions=[condition_gte],
        if_steps=[register_model_step],
        else_steps=[fail_step],
    )
    return condition_step, accuracy_threshold


def create_a_second_condition():
    condition_step = ConditionStep(
        name="check-model-accuracy",
        conditions=[condition_gte],
        if_steps=[register_model_step, deploy_step],
        else_steps=[fail_step],
    )
    return condition_step


def create_a_third_condition():
    condition_step = ConditionStep(
        name="check-model-accuracy",
        conditions=[condition_gte],
        if_steps=[
            create_model_step,
            generate_test_predictions_step,
            model_quality_baseline_step,
            register_model_step,
            deploy_step,
        ],
        else_steps=[fail_step],
    )
    return condition_step
