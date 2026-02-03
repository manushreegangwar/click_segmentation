import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
import fiftyone.zoo as foz


class SaveKeypoints(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="save_keypoints",
            label="Save Keypoints",
            unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("sample_id", required=True, label="Sample ID")
        inputs.list(
            "keypoints",
            types.List(types.Number()),
            required=True,
            label="Keypoints",
        )
        inputs.str("field_name", default="user_clicks", label="Field Name")
        return types.Property(inputs)

    def execute(self, ctx):
        sample_id = ctx.params["sample_id"]
        keypoints = ctx.params["keypoints"]
        field_name = ctx.params["field_name"]

        dataset = ctx.dataset
        sample = dataset[sample_id]

        if sample.has_field(field_name) and sample[field_name] is not None:
            num_kpts = len(sample[field_name].keypoints)
            ctx.ops.notify(
                f"Appending keypoints to {field_name} with {num_kpts} existing keypoints.",
                variant="warning",
            )
            keypoint = fo.Keypoint(points=keypoints, label=f"click_{num_kpts}")
            sample[field_name].keypoints.append(keypoint)
        else:
            keypoint = fo.Keypoint(points=keypoints, label="click_0")
            sample[field_name] = fo.Keypoints(keypoints=[keypoint])
        sample.save()


class SegmentWithKeypoints(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="segment_with_keypoints",
            label="Segment With Keypoints",
            unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("sample_id", required=True, label="Sample ID")
        inputs.str(
            "keypoints_field",
            required=True,
            label="Keypoints Field",
        )
        inputs.str("model_name", required=True, label="Model Name")
        return types.Property(inputs)

    def execute(self, ctx):
        sample_id = ctx.params["sample_id"]
        keypoints_field = ctx.params["keypoints_field"]
        model_name = ctx.params["model_name"]

        model = foz.load_zoo_model(model_name)
        sample_view = ctx.dataset.select(sample_id)

        label_field = keypoints_field + "_seg"
        sample_view.apply_model(
            model,
            label_field=label_field,
            prompt_field=keypoints_field,
        )
        sample_view.save()

        ctx.ops.notify(
            f"Segmentation saved to {label_field}",
            variant="success",
        )


def register(p):
    p.register(SaveKeypoints)
    p.register(SegmentWithKeypoints)
