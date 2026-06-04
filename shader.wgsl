struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VertexOutput {
    var out: VertexOutput;

    var positions = array(
        vec2f(-1.0, -1.0),
        vec2f( 1.0, -1.0),
        vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0),
        vec2f( 1.0, -1.0),
        vec2f( 1.0,  1.0),
    );

    let xy = positions[index];
    out.position = vec4f(xy, 0.0, 1.0);
    out.uv = xy * 0.5 + vec2f(0.5, 0.5);

    return out;
}

@group(0) @binding(0) var t_diffuse: texture_2d<f32>;
@group(0) @binding(1) var s_diffuse: sampler;

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4f {
    return textureSample(t_diffuse, s_diffuse, in.uv);
}
