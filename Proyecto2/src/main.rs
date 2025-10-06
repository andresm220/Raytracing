use raylib::prelude::*;

// ---------------- math util ----------------
#[derive(Clone, Copy, Debug)]
struct Vec3 { x: f32, y: f32, z: f32 }
impl Vec3 {
    fn new(x:f32,y:f32,z:f32)->Self{Self{x,y,z}}
    fn add(self,o:Self)->Self{Self::new(self.x+o.x,self.y+o.y,self.z+o.z)}
    fn sub(self,o:Self)->Self{Self::new(self.x-o.x,self.y-o.y,self.z-o.z)}
    fn mul(self,s:f32)->Self{Self::new(self.x*s,self.y*s,self.z*s)}
    fn had(self,o:Self)->Self{Self::new(self.x*o.x,self.y*o.y,self.z*o.z)}
    fn dot(self,o:Self)->f32{self.x*o.x+self.y*o.y+self.z*o.z}
    fn length(self)->f32{self.dot(self).sqrt()}
    fn norm(self)->Self{let l=self.length(); if l>0.0{self.mul(1.0/l)}else{self}}
    fn cross(self,o:Self)->Self{Self::new(self.y*o.z - self.z*o.y, self.z*o.x - self.x*o.z, self.x*o.y - self.y*o.x)}
}
fn clamp01(x:f32)->f32{x.max(0.0).min(1.0)}
fn mix(a:Vec3,b:Vec3,t:f32)->Vec3{a.mul(1.0-t).add(b.mul(t))}
fn to_color(v:Vec3)->Color{
    let e = 1.15;
    let vx = clamp01(v.x*e).powf(1.0/2.2);
    let vy = clamp01(v.y*e).powf(1.0/2.2);
    let vz = clamp01(v.z*e).powf(1.0/2.2);
    Color::new((vx*255.0) as u8,(vy*255.0) as u8,(vz*255.0) as u8,255)
}

// --------------- cámara orbital ---------------
struct Orbit { target: Vec3, dist: f32, yaw: f32, pitch: f32 }
impl Orbit {
    fn new()->Self{ Self{ target:Vec3::new(0.0,1.5,0.0), dist:12.0, yaw:0.0, pitch:0.35 } }
    fn pos(&self)->Vec3{
        let x=self.dist*self.yaw.cos()*self.pitch.cos();
        let y=self.dist*self.pitch.sin();
        let z=self.dist*self.yaw.sin()*self.pitch.cos();
        self.target.add(Vec3::new(x,y,z))
    }
}

// --------------- rayos/intersección ---------------
#[derive(Clone,Copy)] struct Ray{ o:Vec3, d:Vec3 }
#[derive(Clone,Copy)] struct Hit{ t:f32, p:Vec3, n:Vec3, mat_id:usize, uv:(f32,f32) }

// Textura CPU con bilinear
#[derive(Clone)]
struct CpuTex { w:i32, h:i32, data: Vec<Vec3> }
impl CpuTex {
    fn sample(&self, uv:(f32,f32))->Vec3{ self.sample_bilinear(uv) }
    fn sample_bilinear(&self, uv:(f32,f32))->Vec3{
        let mut u = uv.0.fract(); if u<0.0{u+=1.0;}
        let mut v = uv.1.fract(); if v<0.0{v+=1.0;}
        let x = u * (self.w as f32 - 1.0);
        let y = v * (self.h as f32 - 1.0);
        let x0 = x.floor() as i32; let x1 = (x0+1).min(self.w-1);
        let y0 = y.floor() as i32; let y1 = (y0+1).min(self.h-1);
        let tx = x - x0 as f32; let ty = y - y0 as f32;
        let idx = |xx:i32, yy:i32| -> usize { (yy*self.w + xx) as usize };
        let c00 = self.data[idx(x0,y0)];
        let c10 = self.data[idx(x1,y0)];
        let c01 = self.data[idx(x0,y1)];
        let c11 = self.data[idx(x1,y1)];
        let cx0 = mix(c00, c10, tx);
        let cx1 = mix(c01, c11, tx);
        mix(cx0, cx1, ty)
    }
}

#[derive(Clone)]
enum TextureKind {
    None,
    Checker{scale:f32, dark:Vec3, light:Vec3},
    Stripes{scale:f32, a:Vec3, b:Vec3},
    Image(CpuTex),
    GradientV{ top:Vec3, bottom:Vec3 },
    Bricks { scale:f32, mortar:Vec3, brick_a:Vec3, brick_b:Vec3 },
    NoiseSoft { scale:f32, a:Vec3, b:Vec3 },
}

#[derive(Clone,Copy)]
struct Material {
    albedo: Vec3,
    specular: f32,
    reflectivity: f32,
    transparency: f32,
    ior: f32,
}

// Piso como AABB
#[derive(Clone)]
struct BoxAA { min:Vec3, max:Vec3, mat_id:usize, tex_id:usize }
impl BoxAA {
    fn intersect(&self, r:Ray)->Option<(f32,Vec3,Vec3,(f32,f32))>{
        let inv=Vec3::new(1.0/r.d.x,1.0/r.d.y,1.0/r.d.z);
        let mut tx1=(self.min.x-r.o.x)*inv.x; let mut tx2=(self.max.x-r.o.x)*inv.x;
        let mut tmin=tx1.min(tx2); let mut tmax=tx1.max(tx2);
        let ty1=(self.min.y-r.o.y)*inv.y; let ty2=(self.max.y-r.o.y)*inv.y;
        tmin=tmin.max(ty1.min(ty2)); tmax=tmax.min(ty1.max(ty2));
        let tz1=(self.min.z-r.o.z)*inv.z; let tz2=(self.max.z-r.o.z)*inv.z;
        tmin=tmin.max(tz1.min(tz2)); tmax=tmax.min(tz1.max(tz2));
        if tmax>=tmin.max(0.001){
            let t= if tmin>=0.001{tmin}else{tmax};
            let p=r.o.add(r.d.mul(t));
            let eps=1e-4;
            let (n,uv) = if (p.x-self.min.x).abs()<eps {
                let u=(p.z-self.min.z)/(self.max.z-self.min.z);
                let v=(p.y-self.min.y)/(self.max.y-self.min.y);
                (Vec3::new(-1.0,0.0,0.0),(u,v))
            } else if (p.x-self.max.x).abs()<eps {
                let u=1.0-(p.z-self.min.z)/(self.max.z-self.min.z);
                let v=(p.y-self.min.y)/(self.max.y-self.min.y);
                (Vec3::new(1.0,0.0,0.0),(u,v))
            } else if (p.y-self.min.y).abs()<eps {
                let u=(p.x-self.min.x)/(self.max.x-self.min.x);
                let v=(p.z-self.min.z)/(self.max.z-self.min.z);
                (Vec3::new(0.0,-1.0,0.0),(u,v))
            } else if (p.y-self.max.y).abs()<eps {
                let u=(p.x-self.min.x)/(self.max.x-self.min.x);
                let v=1.0-(p.z-self.min.z)/(self.max.z-self.min.z);
                (Vec3::new(0.0,1.0,0.0),(u,v))
            } else if (p.z-self.min.z).abs()<eps {
                let u=(p.x-self.min.x)/(self.max.x-self.min.x);
                let v=(p.y-self.min.y)/(self.max.y-self.min.y);
                (Vec3::new(0.0,0.0,-1.0),(u,v))
            } else {
                let u=1.0-(p.x-self.min.x)/(self.max.x-self.min.x);
                let v=(p.y-self.min.y)/(self.max.y-self.min.y);
                (Vec3::new(0.0,0.0,1.0),(u,v))
            };
            Some((t,p,n,uv))
        } else { None }
    }
}

// Cubo rotado (OBB) con yaw
#[derive(Clone)]
struct Cube {
    center: Vec3,
    half:   Vec3,
    yaw:    f32,
    mat_id: usize,
    tex_id: usize,
}
fn rot_y(v:Vec3, a:f32)->Vec3{
    let (s,c) = (a.sin(), a.cos());
    Vec3::new(c*v.x + s*v.z, v.y, -s*v.x + c*v.z)
}
impl Cube {
    fn intersect(&self, r:Ray) -> Option<(f32,Vec3,Vec3,(f32,f32))>{
        let ro_local = rot_y(r.o.sub(self.center), -self.yaw);
        let rd_local = rot_y(r.d, -self.yaw);
        let inv = Vec3::new(1.0/rd_local.x, 1.0/rd_local.y, 1.0/rd_local.z);
        let min = self.half.mul(-1.0);
        let max = self.half;
        let mut tx1=(min.x-ro_local.x)*inv.x; let mut tx2=(max.x-ro_local.x)*inv.x;
        let mut tmin=tx1.min(tx2); let mut tmax=tx1.max(tx2);
        let ty1=(min.y-ro_local.y)*inv.y; let ty2=(max.y-ro_local.y)*inv.y;
        tmin=tmin.max(ty1.min(ty2)); tmax=tmax.min(ty1.max(ty2));
        let tz1=(min.z-ro_local.z)*inv.z; let tz2=(max.z-ro_local.z)*inv.z;
        tmin=tmin.max(tz1.min(tz2)); tmax=tmax.min(tz1.max(tz2));
        if tmax>=tmin.max(0.001){
            let t_local = if tmin>=0.001{tmin}else{tmax};
            let p_local = ro_local.add(rd_local.mul(t_local));
            let eps=1e-4;
            let (n_local, uv) = if (p_local.x - min.x).abs()<eps {
                let u=(p_local.z - min.z)/(max.z-min.z);
                let v=(p_local.y - min.y)/(max.y-min.y);
                (Vec3::new(-1.0,0.0,0.0), (u,v))
            } else if (p_local.x - max.x).abs()<eps {
                let u=1.0-(p_local.z - min.z)/(max.z-min.z);
                let v=(p_local.y - min.y)/(max.y-min.y);
                (Vec3::new(1.0,0.0,0.0), (u,v))
            } else if (p_local.y - min.y).abs()<eps {
                let u=(p_local.x - min.x)/(max.x-min.x);
                let v=(p_local.z - min.z)/(max.z-min.z);
                (Vec3::new(0.0,-1.0,0.0), (u,v))
            } else if (p_local.y - max.y).abs()<eps {
                let u=(p_local.x - min.x)/(max.x-min.x);
                let v=1.0-(p_local.z - min.z)/(max.z-min.z);
                (Vec3::new(0.0,1.0,0.0), (u,v))
            } else if (p_local.z - min.z).abs()<eps {
                let u=(p_local.x - min.x)/(max.x-min.x);
                let v=(p_local.y - min.y)/(max.y-min.y);
                (Vec3::new(0.0,0.0,-1.0), (u,v))
            } else {
                let u=1.0-(p_local.x - min.x)/(max.x-min.x);
                let v=(p_local.y - min.y)/(max.y-min.y);
                (Vec3::new(0.0,0.0,1.0), (u,v))
            };
            let p_world = self.center.add(rot_y(p_local, self.yaw));
            let n_world = rot_y(n_local, self.yaw).norm();
            Some((t_local, p_world, n_world, uv))
        } else { None }
    }
}

// --------------- escena ---------------
struct Scene {
    floor: BoxAA,
    cubes: Vec<Cube>,
    mats:  Vec<Material>,
    texs:  Vec<TextureKind>,
    light_dir: Vec3,
    ambient: Vec3,
}
impl Scene {
    fn sample_sky(&self, d:Vec3)->Vec3{
        let t = 0.5*(d.y+1.0);
        let top = Vec3::new(0.55,0.65,0.85);
        let bot = Vec3::new(0.85,0.90,1.00);
        mix(bot, top, t)
    }
    fn shade_albedo(&self, tex_id:usize, uv:(f32,f32))->Vec3{
        match &self.texs[tex_id]{
            TextureKind::None => Vec3::new(1.0,1.0,1.0),
            TextureKind::Checker{scale,dark,light}=>{
                let s=*scale; let u=(uv.0*s).floor() as i32; let v=(uv.1*s).floor() as i32;
                if ((u+v)&1)==1 { *dark } else { *light }
            }
            TextureKind::Stripes{scale,a,b}=>{
                let s=*scale; let t=(((uv.0*s).sin()).signum()+1.0)*0.5; a.mul(1.0-t).add(b.mul(t))
            }
            TextureKind::Image(img)=> img.sample(uv),
            TextureKind::GradientV{top,bottom}=>{
                let v = clamp01(uv.1);
                mix(*bottom, *top, v)
            }
            TextureKind::Bricks{scale, mortar, brick_a, brick_b}=>{
                let s=*scale;
                let u = uv.0 * s;
                let v = uv.1 * s;
                let row = v.floor() as i32;
                let u2 = if row & 1 == 0 { u } else { u + 0.5 };
                let fu = u2.fract();
                let fv = v.fract();
                let mortar_w = 0.06;
                let is_mortar = fu < mortar_w || fv < mortar_w;
                if is_mortar { *mortar } else {
                    if ((u2.floor() as i32 + row) & 1) == 0 { *brick_a } else { *brick_b }
                }
            }
            TextureKind::NoiseSoft{scale, a, b}=>{
                let s=*scale;
                let n = ((uv.0*s).sin()*1.3 + (uv.1*s*1.7).cos()*0.9
                    + ((uv.0+uv.1)*s*0.5).sin()*0.6) * 0.25 + 0.5;
                mix(*a, *b, clamp01(n))
            }
        }
    }
    fn intersect(&self, r:Ray)->Option<Hit>{
        let mut best = f32::INFINITY;
        let mut out: Option<Hit> = None;
        if let Some((t,p,n,uv)) = self.floor.intersect(r) {
            if t<best && t>0.001 { best=t; out=Some(Hit{t,p,n,mat_id:self.floor.mat_id,uv}); }
        }
        for c in &self.cubes {
            if let Some((t,p,n,uv)) = c.intersect(r) {
                if t<best && t>0.001 { best=t; out=Some(Hit{t,p,n,mat_id:c.mat_id,uv}); }
            }
        }
        out
    }
}

// --------------- óptica ---------------
fn reflect(i:Vec3,n:Vec3)->Vec3 { i.sub(n.mul(2.0*i.dot(n))) }
fn refract(i:Vec3, n:Vec3, ior:f32)->Option<Vec3>{
    let cosi = -(i.dot(n)).clamp(-1.0,1.0);
    let (n, eta, cosi) = if cosi<0.0 { (n.mul(-1.0), 1.0/ior, -cosi) } else { (n, ior, cosi) };
    let k = 1.0 - eta*eta*(1.0 - cosi*cosi);
    if k<0.0 { None } else { Some(i.mul(eta).add(n.mul(eta*cosi - k.sqrt()))) }
}
fn fresnel(i:Vec3,n:Vec3,ior:f32)->f32{
    let cosi = clamp01(-(i.dot(n)).abs());
    let etai=1.0; let etat=ior;
    let sint = etai/etat*(1.0-cosi*cosi).sqrt();
    if sint>=1.0 { return 1.0; }
    let cost = (1.0 - sint*sint).sqrt();
    let rs = ((etat*cosi)-(etai*cost))/((etat*cosi)+(etai*cost));
    let rp = ((etai*cosi)-(etat*cost))/((etai*cosi)+(etat*cost));
    (rs*rs + rp*rp) * 0.5
}

// --------------- RT config ---------------
// resolución dinámica
#[derive(Clone, Copy)]
struct RtSize { w: i32, h: i32 }
const LOW:  RtSize = RtSize{ w: 640, h: 360 };
const MED:  RtSize = RtSize{ w: 800, h: 450 };
const HIGH: RtSize = RtSize{ w: 900, h: 506 };

const MAX_DEPTH:i32=3; // baja un poco para más FPS

// Sombrador
fn shade(scene:&Scene, r:Ray, depth:i32)->Vec3{
    if depth>MAX_DEPTH { return Vec3::new(0.0,0.0,0.0); }
    if let Some(h)=scene.intersect(r){
        let m = scene.mats[h.mat_id];

        if h.mat_id <= 4 {
            let base = scene.shade_albedo(h.mat_id, h.uv).had(m.albedo);
            if h.mat_id == 4 {
                let alpha = 125.0/255.0; // vidrio como el primero
                let behind = shade(scene, Ray { o: h.p.add(r.d.mul(0.002)), d: r.d }, depth+1);
                return base.mul(alpha).add(behind.mul(1.0 - alpha));
            }
            return base;
        }

        // Piso agua
        let base = scene.shade_albedo(h.mat_id, h.uv).had(m.albedo);
        let n = h.n; 
        let p = h.p.add(n.mul(0.001));
        let mut col = base;
        let kr = fresnel(r.d, n, m.ior);

        if m.reflectivity > 0.0 {
            let rdir = reflect(r.d, n).norm();
            let rcol = shade(scene, Ray { o: p, d: rdir }, depth+1);
            col = mix(col, rcol, m.reflectivity * kr);
        }
        if m.transparency > 0.0 {
            if let Some(tdir) = refract(r.d, n, m.ior) {
                let tcol = shade(scene, Ray { o: h.p.sub(n.mul(0.001)), d: tdir.norm() }, depth+1);
                col = mix(col, tcol, m.transparency * (1.0 - kr));
            }
        }
        return col;
    } else {
        scene.sample_sky(r.d)
    }
}

fn main(){
    let (mut rl, thread) = raylib::init()
        .size(900,600)
        .title("Diorama estilo Minecraft: rombo flotante + agua + vidrio (alpha)")
        .build();
    rl.set_target_fps(60);

    // PNGs -> CPU
    let img_madera = Image::load_image("assets/caja_madera.png").expect("Falta assets/caja_madera.png");
    let img_metal  = Image::load_image("assets/caja_metal.png").expect("Falta assets/caja_metal.png");
    let madera_cpu = CpuTex{
        w: img_madera.width(), h: img_madera.height(),
        data: img_madera.get_image_data().iter()
            .map(|c| Vec3::new(c.r as f32/255.0, c.g as f32/255.0, c.b as f32/255.0)).collect()
    };
    let metal_cpu = CpuTex{
        w: img_metal.width(), h: img_metal.height(),
        data: img_metal.get_image_data().iter()
            .map(|c| Vec3::new(c.r as f32/255.0, c.g as f32/255.0, c.b as f32/255.0)).collect()
    };

    // Materiales
    let mats = vec![
        Material{ albedo:Vec3::new(1.0,1.0,1.0), specular:0.0, reflectivity:0.06, transparency:0.0,  ior:1.0 }, // 0 madera
        Material{ albedo:Vec3::new(1.0,1.0,1.0), specular:0.0, reflectivity:0.06, transparency:0.0,  ior:1.0 }, // 1 grass
        Material{ albedo:Vec3::new(1.0,1.0,1.0), specular:0.05, reflectivity:0.08, transparency:0.0,  ior:1.0 }, // 2 cobble
        Material{ albedo:Vec3::new(1.0,1.0,1.0), specular:0.0, reflectivity:0.10, transparency:0.0,  ior:1.0 }, // 3 metal
        Material{ albedo:Vec3::new(1.0,1.0,1.0), specular:0.0, reflectivity:0.00, transparency:0.0,  ior:1.0 }, // 4 vidrio
        Material{ albedo:Vec3::new(0.85,0.95,1.0),specular:0.8, reflectivity:0.45, transparency:0.75, ior:1.33 }, // 5 agua
    ];

    // Texturas
    let texs = vec![
        TextureKind::Image(madera_cpu), // 0 madera
        TextureKind::NoiseSoft{ // 1 GRASS
            scale: 22.0,
            a: Vec3::new(0.20, 0.45, 0.18),
            b: Vec3::new(0.32, 0.65, 0.28),
        },
        TextureKind::Bricks{ // 2 COBBLESTONE
            scale: 10.0,
            mortar:  Vec3::new(0.35,0.37,0.40),
            brick_a: Vec3::new(0.55,0.57,0.60),
            brick_b: Vec3::new(0.50,0.52,0.55),
        },
        TextureKind::Image(metal_cpu), // 3 metal
        TextureKind::GradientV{ // 4 vidrio (igual al primero)
            top:    Vec3::new(240.0/255.0, 255.0/255.0, 255.0/255.0),
            bottom: Vec3::new(200.0/255.0, 230.0/255.0, 255.0/255.0),
        },
        TextureKind::NoiseSoft{ // 5 agua
            scale: 8.0,
            a: Vec3::new(0.12,0.25,0.45),
            b: Vec3::new(0.06,0.15,0.30),
        },
    ];

    // Agua plana en y=0
    let floor = BoxAA{
        min: Vec3::new(-60.0, -0.02, -60.0),
        max: Vec3::new( 60.0,  0.00,  60.0),
        mat_id: 5, tex_id: 5
    };

    // Rombo + centro
    let r = 3.0;
    let centers = [
        Vec3::new(-r, 1.5,  0.0), // madera (izq)
        Vec3::new( 0.0, 1.8,  r), // grass (arriba)
        Vec3::new( r, 1.3,  0.0), // cobble (der)
        Vec3::new( 0.0, 1.6, -r), // metal (abajo)
        Vec3::new( 0.0, 2.0,  0.0), // vidrio (centro)
    ];

    // Cubos
    let mut cubes = vec![
        Cube{ center:centers[0], half:Vec3::new(0.5,0.5,0.5), yaw:0.0, mat_id:0, tex_id:0 },
        Cube{ center:centers[1], half:Vec3::new(0.5,0.5,0.5), yaw:0.0, mat_id:1, tex_id:1 },
        Cube{ center:centers[2], half:Vec3::new(0.5,0.5,0.5), yaw:0.0, mat_id:2, tex_id:2 },
        Cube{ center:centers[3], half:Vec3::new(0.5,0.5,0.5), yaw:0.0, mat_id:3, tex_id:3 },
        Cube{ center:centers[4], half:Vec3::new(0.55,0.55,0.55), yaw:0.0, mat_id:4, tex_id:4 },
    ];
    let spin = [0.6_f32, -0.45, 0.8, -0.5, 0.7];

    let mut scene = Scene{
        floor,
        cubes,
        mats, texs,
        light_dir: Vec3::new(-0.3,-1.0,-0.15),
        ambient: Vec3::new(0.10,0.10,0.11),
    };

    // ---- cámara / input ----
    let mut orb = Orbit::new();
    let mut yaw = 0.0f32;
    let mut pitch = 0.35f32;
    let sens=0.01f32; let zoom_sens=1.2f32;
    let mut last_mouse = rl.get_mouse_position(); let mut dragging=false;

    // ---- resolución dinámica ----
    let mut rt = MED; // arranca MED
    let mut frame: Vec<u8> = vec![0; (rt.w*rt.h*4) as usize];
    let mut tex = rl
        .load_texture_from_image(&thread, &Image::gen_image_color(rt.w, rt.h, Color::BLACK))
        .unwrap();

    while !rl.window_should_close(){
        // hotkeys de calidad
        if rl.is_key_pressed(KeyboardKey::KEY_ONE)   { rt = LOW; }
        if rl.is_key_pressed(KeyboardKey::KEY_TWO)   { rt = MED; }
        if rl.is_key_pressed(KeyboardKey::KEY_THREE) { rt = HIGH; }
        // recreate buffers si cambió
        if (frame.len() as i32) != (rt.w*rt.h*4) {
            frame = vec![0; (rt.w*rt.h*4) as usize];
            tex = rl
                .load_texture_from_image(&thread, &Image::gen_image_color(rt.w, rt.h, Color::BLACK))
                .unwrap();
        }

        let dt = rl.get_frame_time();

        // input cámara (igual que antes, con zoom/deszoom en rueda)
        let mouse=rl.get_mouse_position();
        let is_down=rl.is_mouse_button_down(MouseButton::MOUSE_LEFT_BUTTON);
        let mut dx=0.0; let mut dy=0.0;
        if is_down && dragging { dx=mouse.x-last_mouse.x; dy=mouse.y-last_mouse.y; }
        last_mouse=mouse; dragging=is_down;

        yaw += dx*sens; pitch -= dy*sens;
        let max_pitch=1.55; if pitch> max_pitch {pitch=max_pitch;} if pitch< -max_pitch {pitch=-max_pitch;}
        let wheel=rl.get_mouse_wheel_move();
        if wheel>0.0 { orb.dist/=zoom_sens; }
        if wheel<0.0 { orb.dist*=zoom_sens; }
        orb.dist=orb.dist.clamp(3.0, 40.0);
        orb.yaw=yaw; orb.pitch=pitch;

        // rotación por cubo
        for (i,c) in scene.cubes.iter_mut().enumerate() { c.yaw += spin[i] * dt; }

        // cámara pinhole
        let eye = orb.pos(); let target=orb.target;
        let forward=target.sub(eye).norm();
        let right=forward.cross(Vec3::new(0.0,1.0,0.0)).norm();
        let up   = right.cross(forward).norm();
        let fov = 60.0_f32.to_radians();
        let aspect = rt.w as f32 / rt.h as f32;
        let scale=(fov*0.5).tan();

        // trazar en el RT actual
        for y in 0..rt.h {
            for x in 0..rt.w {
                let u=((x as f32+0.5)/rt.w as f32)*2.0-1.0;
                let v=(1.0-(y as f32+0.5)/rt.h as f32)*2.0-1.0;
                let dir = forward.add(right.mul(u*aspect*scale)).add(up.mul(v*scale)).norm();
                let col = shade(&scene, Ray{o:eye,d:dir}, 0);
                let idx=((y*rt.w+x)*4) as usize;
                let c = to_color(col);
                frame[idx]=c.r; frame[idx+1]=c.g; frame[idx+2]=c.b; frame[idx+3]=255;
            }
        }

        tex.update_texture(&frame);

        // draw (escalado a la ventana)
        let mut d=rl.begin_drawing(&thread);
        d.clear_background(Color::new(8,9,12,255));
        let win_w=d.get_screen_width(); let win_h=d.get_screen_height();
        let sx=win_w as f32/rt.w as f32; let sy=win_h as f32/rt.h as f32; let s=sx.min(sy);
        let dw=(rt.w as f32*s) as i32; let dh=(rt.h as f32*s) as i32; let dx=(win_w-dw)/2; let dy=(win_h-dh)/2;
        d.draw_texture_pro(
            &tex,
            Rectangle{ x:0.0,y:0.0,width:rt.w as f32,height:rt.h as f32 },
            Rectangle{ x:dx as f32,y:dy as f32,width:dw as f32,height:dh as f32 },
            Vector2::new(0.0,0.0), 0.0, Color::WHITE
        );
        d.draw_text("Arrastra: orbitar | Rueda: zoom | 1/2/3: LOW/MED/HIGH | Agua (refl+refr) | Centro=Vidrio", 10, 10, 18, Color::RAYWHITE);
        d.draw_fps(10, 34);
    }
}
