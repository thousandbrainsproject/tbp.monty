---
title: Architecture Overview
---
There are three major components that play a role in the architecture: sensors, learning modules, and actuators [\[1\]](#footnote1). These three components are tied together by a common messaging protocol, which we call the cortical messaging protocol (CMP). Due to the unified messaging protocol, the inner workings of each individual component can be quite varied as long as they have the appropriate interfaces [\[2\]](#footnote2).

Those three components and the CMP are described in the following sub-sections. For a presentation of all the content in this sections (+a few others), have a look at the recording from our launch symposium:

[block:embed]
{
  "html": "<iframe class=\"embedly-embed\" src=\"//cdn.embedly.com/widgets/media.html?src=https%3A%2F%2Fwww.youtube.com%2Fembed%2FlqFZKlsb8Dc%3Ffeature%3Doembed&display_name=YouTube&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DlqFZKlsb8Dc&image=https%3A%2F%2Fi.ytimg.com%2Fvi%2FlqFZKlsb8Dc%2Fhqdefault.jpg&type=text%2Fhtml&schema=youtube\" width=\"854\" height=\"480\" scrolling=\"no\" title=\"YouTube embed\" frameborder=\"0\" allow=\"autoplay; fullscreen; encrypted-media; picture-in-picture;\" allowfullscreen=\"true\"></iframe>",
  "url": "https://www.youtube.com/watch?v=lqFZKlsb8Dc",
  "title": "2024/12 Overview of the TBP and the Monty Implementation",
  "favicon": "https://www.youtube.com/favicon.ico",
  "image": "https://i.ytimg.com/vi/lqFZKlsb8Dc/hqdefault.jpg",
  "provider": "https://www.youtube.com/",
  "href": "https://www.youtube.com/watch?v=lqFZKlsb8Dc",
  "typeOfEmbed": "youtube"
}
[/block]

## Footnotes

<a name="footnote1">1</a>: Sensors may be actuators and could have capabilities to take a motor command to move or attend to a new location.

<a name="footnote2">2</a>: In general, the learning modules in an instance of Monty will adhere to the concepts described herein, however it is possible to augment Monty with alternative learning modules. For example, we do _not_ anticipate that the learning modules described herein will be useful for calculating the result of numerical functions, or for predicting the structure of a protein given its genetic sequence. Alternative systems could therefore be leveraged for such tasks and then interfaced according to the CMP.