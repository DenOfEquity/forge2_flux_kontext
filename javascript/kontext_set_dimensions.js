function kontext_set_dimensions(tab, dims) {
	if (!dims || dims == '0') {
		return;
	}

	if (tab === 'txt2img') {
		width_component = gradioApp().getElementById('txt2img_width');
		height_component = gradioApp().getElementById('txt2img_height');
	}
	else if (tab === 'img2img') {
		width_component = gradioApp().getElementById('img2img_width');
		height_component = gradioApp().getElementById('img2img_height');
	}
	else {
		return;
	}

	d = dims.split(',');
	w = parseInt(d[0], 10);
	h = parseInt(d[1], 10);

	width_component = width_component.querySelector('div > div > input');
	height_component = height_component.querySelector('div > div > input');

	width_component.value = w;
	height_component.value = h;

	updateInput(width_component);
	updateInput(height_component);

	return;
}
