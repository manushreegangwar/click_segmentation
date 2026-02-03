import { PluginComponentType, registerComponent } from "@fiftyone/plugins";
import { ClickSegmentation } from "./ClickSegmentation";

registerComponent({
  name: "ClickSegmentation",
  label: "Click To Segment",
  component: ClickSegmentation,
  type: PluginComponentType.Panel,
  activator: myActivator,
  panelOptions: {
    surfaces: "modal",
  },
});

function myActivator({ dataset }) {
  // Example of activating the plugin in a particular context
  // return dataset.name === 'quickstart'

  return true;
}
