// Embeds the app icon (installer/icon/rvc-app.ico) in rvc-app.exe for Explorer, the taskbar and shortcuts.
fn main() {
    const ICON: &str = "../../installer/icon/rvc-app.ico";
    println!("cargo:rerun-if-changed={ICON}");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon(ICON);
        res.set("ProductName", "RVC Realtime");
        res.set("FileDescription", "RVC Realtime");
        res.compile().expect("embed the app icon");
    }
}
