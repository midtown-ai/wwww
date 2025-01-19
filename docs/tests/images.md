# Images


## Relative path

![](../glossary/img/a/activation_function.png){style="width:40%"}

```
![](../glossary/img/a/activation_function.png){style="width:40%"}
```

## Absolute path

![](/glossary/img/a/activation_function.png){style="width:40%"}

```
![](/glossary/img/a/activation_function.png){style="width:40%"}
```

## Caption

![](../glossary/img/a/activation_function.png){style="width:40%"}
/// caption
Image and caption are automatically centered!
///

```
![](../glossary/img/a/activation_function.png){style="width:40%"}
/// caption
Image and caption are automatically centered!
///
```

## Dark vs light mode (CSS)

### raw HTML

<div class="theme-image">
  <img src="../img/zelda_light_mode.png" alt="Light Mode Image" class="light-mode" style="width:40%">
  <img src="../img/zelda_dark_mode.png" alt="Dark Mode Image" class="dark-mode" style="width:40%">
</div>

```
# note the path to the image files is 1 directory up!

<div class="theme-image">
  <img src="../img/zelda_light_mode.png" alt="Light Mode Image" class="light-mode">
  <img src="../img/zelda_dark_mode.png" alt="Dark Mode Image" class="dark-mode">
</div>

# CSS selector

.theme-image img {
    /* Styles for all images inside .theme-image */
}

# or to select a particular image in the div

.theme-image .light-mode {
    /* Styles for the light mode image */
}
```


### with markdown

![](img/zelda_light_mode.png){ .light-mode style="width:40%"}
![](img/zelda_dark_mode.png){ .dark-mode style="width:40%"}

```
# markdown

![](img/zelda_light_mode.png){ .light-mode style="width:40%"}
![](img/zelda_dark_mode.png){ .dark-mode style="width:40%"}

# to html

<p><img alt="" class="light-mode" src="../img/zelda_light_mode.png" style="width:40%">
<img alt="" class="dark-mode" src="../img/zelda_dark_mode.png" style="width:40%"></p></div>
 
# CSS selector

img.light-mode {
    /* Your styles here */
}

```

![](img/zelda_light_mode.png){ .theme-image .light-mode style="width:40%"}
![](img/zelda_dark_mode.png){ .theme-image .dark-mode style="width:40%"}


```
# markdown

![](img/zelda_light_mode.png){ .theme-image .light-mode style="width:40%"}
![](img/zelda_dark_mode.png){ .theme-image .dark-mode style="width:40%"}

# to html

<img alt="" class="theme-image light-mode" src="../img/zelda_light_mode.png" style="width:40%">
<img alt="" class="theme-image dark-mode" src="../img/zelda_dark_mode.png" style="width:40%"></p>

# CSS selector

img.theme-image.light-mode {
    /* Images must have BOTH classes to be selected */
}

# or

img.light-mode {
    /* Images must have the light-mode class */
}
```

