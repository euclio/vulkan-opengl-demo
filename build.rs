use std::env;
use std::fs::File;
use std::path::Path;

use gl_generator::{Api, Fallbacks, GlobalGenerator, Profile, Registry};

fn main() {
    let dest = env::var("OUT_DIR").unwrap();
    let mut file = File::create(Path::new(&dest).join("gl_bindings.rs")).unwrap();

    Registry::new(
        Api::Gl,
        (3, 0),
        Profile::Compatibility,
        Fallbacks::All,
        [
            "GL_EXT_memory_object",
            "GL_EXT_memory_object_fd",
            "GL_EXT_semaphore",
            "GL_EXT_semaphore_fd",
        ],
    )
    .write_bindings(GlobalGenerator, &mut file)
    .unwrap();
}
