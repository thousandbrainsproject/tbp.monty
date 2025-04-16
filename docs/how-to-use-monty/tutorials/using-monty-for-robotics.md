---
title: Using Monty for Robotics
---
# Introduction
As Monty is a sensorimotor learning system, robotics is one big field of applications that naturally comes to mind. This tutorial explains in more detail how Monty can be used for robotics applications outside a simulator. It builds on the [previous tutorial](./using-monty-in-a-custom-application.md), so if you haven't read it yet, we highly recommend starting there.

# Monty <-> Robot Interface
Currently, Monty relies on a couple of dependencies that [can NOT be installed on standard robotics hardware, such as the Raspberry Pi](https://github.com/thousandbrainsproject/tbp.monty/issues/85). We are working on removing those from the default dependencies, but for now, we recommend not to run Monty on the Robot's hardware directly. Instead, one can **stream the sensor outputs and action commands back and forth between the physical hardware and Monty running on a laptop** (or other cloud computing infrastructure). This has the advantage of simplifying debugging and visualizing what happens in Monty. It also makes it easier to run more complex instances of Monty (many learning modules) without running into the limitations of on-device computational power. For some applications, the additional delay of streaming the data may cause issues, and future work will investigate how big of a problem this is and how we can allow Monty to run on the device in those cases.

# Example 1: Monty Meets World iPad App
Monty Meets World is the code name for our first demo of Monty on real-world data. For a video of this momentous moment, see our [project showcase page](https://thousandbrainsproject.readme.io/docs/project-showcase#monty-for-object-detection-with-the-ipad-camera).

In this application, we wrote a [MontyMeetsWorld iOS app](https://github.com/thousandbrainsproject/monty_lab/tree/main/monty_meets_world ) that runs locally on the iPad (or iPhone). The app has a button that the user can press to take an image with the user-facing TrueDepth camera. When a picture is taken, it is streamed to a server running locally on a laptop, where it gets saved in a dedicated folder. At the same time, Monty is running on the laptop.
![Data is recorded from a physical sensor (iPad camera) and streamed to a laptop. The Monty instance runs on the laptop and uses the streamed data to recognize the object.](../../figures/how-to-use-monty/streaming_data_to_monty.png)

Monty is configured to use the `SaccadeOnImageFromStreamDataLoader`. The data loader's `pre_epoch` function calls `switch_to_scene` on the `SaccadeOnImageFromSteamEnvironment`, which does nothing until a new image is found in the dedicated folder. Once it detects that a new image was saved there, it loads this image, and the episode starts. The dataloader then moves a small patch over the image (the same way as in the non-streamed version explained in the [previous tutorial](./using-monty-in-a-custom-application.md)) and sends the observations from the moving patch to Monty until Monty recognizes the object. After that, it ends the current episode and returns to waiting for the next image, which will start the next episode.
![Custom classes for object recognition in RGBD images streamed from an iOS app.](../../figures/how-to-use-monty/MMW_stream_custom_classes.png#width=500px)

> 📘 Follow Along
> If you like to test the MontyMeetsWorld app, you can find code and run instructions [here](https://github.com/thousandbrainsproject/monty_lab/tree/main/monty_meets_world). To run the demo there are three main steps involved:
> 1)  Open the MontyMeetsWorld project in XCode and run the iOS app on your iPad or iPhone (instructions in [this README](https://github.com/thousandbrainsproject/monty_lab/blob/main/monty_meets_world/README.md))
> 2) Start a server on your laptop to listen for images streamed from the app by running `python src/tbp/monty/frameworks/environment_utils/server.py`
> 3) Start a Monty experiment that will wait for an image to be received and then run an episode to recognize the object. Use this command in a separate terminal window (while the server script and app are running): `python benchmarks/run.py -e world_image_from_stream_on_scanned_model`
>
> Make sure to set your WiFi's IP address in the server.py script and the app settings on your device. Then, once the app, the server, and the Monty Experiment are running, you can show an object to the camera and press the `Save Image` button in the app.

# Example 2: Ultrasound

> 📘 This Example is Only Conceptual [For Now]
> Note that this example includes no code as we have not yet implemented it. We are planning to build this as one of the demos of Monty during our team-internal [May 2025 Robot Hackathon](https://thousandbrains.discourse.group/t/your-robot-expertise-is-requested/422/27). We will add more details on the projects we built after the hackathon.

For the ultrasound demo project, we went through the same thought process as outlined for any Monty application in the [previous tutorial](./using-monty-in-a-custom-application.md). We needed to define observations, movement, and how movement affects the state of the sensor and its observations. The sensor is a handheld ultrasound device. In this case, Monty is not actively moving the sensor. Instead, a human operator moves the ultrasound device while Monty can (optionally) suggest positions to move the sensor to the operator. Although Monty is not actively moving the sensor, it still needs to know how the sensor is moving. For this, we decided to attach a [Vive Tracker](https://github.com/snuvclab/Vive_Tracker) to the ultrasound device, which uses two wall-mounted base stations to track the pose (6DOF location and orientation) of the sensor in the room.

![Setup for using Monty on data streamed from an iOS app from an ultrasound device and position tracking system.](../../figures/how-to-use-monty/ultrasound_example.png)

Both the location information and the ultrasound recordings are streamed to a laptop that runs Monty. To stream the ultrasound data, we can write a small iOS app, similar to the MontyMeetsWorld app, using the ultrasound device SDK. Once the ultrasound image arrives on the laptop, Monty can move a small patch over it. It can also use the tracked sensor location in the room to integrate the physical movement of the probe.

A custom sensor module can then extract features and poses from the ultrasound image. The pose could be extracted from the point normal detected at borders in the patch. Features could summarize information about measured density and texture.

For the first test, we plan to recognize 3D objects inside a phantom (a clear bag filled with fluid). For simplicity, the objects can be learned beforehand in simulation, similar to the Monty Meets World application. However, since we have 6DOF pose tracking of the sensor, we could also learn the objects directly from the ultrasound data in the real world.

Monty can use three types of action output to recognize the sensed objects efficiently. 1) it can move the patch over the full ultrasound image, akin to moving it over the full camera image from the iPad. 2) it can suggest a location in the room for the human operator to move the ultrasound probe to in order to get a different view. 3) it can adjust the settings of the probe, such as depth of field and gain. None of these are required for object recognition to work, but each one can make recognition more efficient and robust.

# Conclusion
The current solution for running Monty on robots is to stream the sensor data and action commands back and forth between the robot and a Monty instance running on a laptop. Outside of that, defining custom data loader and environments for Monty is analogous to how it was outlined in the [previous tutorial](./using-monty-in-a-custom-application.md).


