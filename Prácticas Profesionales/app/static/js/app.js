document.addEventListener("DOMContentLoaded", function () {
    console.log("Aplicación cargada correctamente.");

    // Resaltar tab activa con clase adicional
    function marcarTabActiva() {
        const tabs = document.querySelectorAll(".nav-tabs .nav-link");
        tabs.forEach(tab => tab.classList.remove("tab-activa-extra"));

        const activa = document.querySelector(".nav-tabs .nav-link.active");
        if (activa) {
            activa.classList.add("tab-activa-extra");
        }
    }

    marcarTabActiva();

    document.addEventListener("click", function (e) {
        if (e.target.classList.contains("nav-link")) {
            setTimeout(marcarTabActiva, 100);
        }
    });

    // Scroll automático suave a resultados de predicción al presionar botón
    document.addEventListener("click", function (e) {
        const texto = (e.target.innerText || "").trim().toLowerCase();
        if (texto === "generar predicción") {
            setTimeout(() => {
                const cards = document.querySelectorAll(".card-header");
                for (const c of cards) {
                    if ((c.innerText || "").toLowerCase().includes("serie histórica y predicción")) {
                        c.scrollIntoView({ behavior: "smooth", block: "start" });
                        break;
                    }
                }
            }, 700);
        }
    });
});

document.addEventListener("DOMContentLoaded", function () {
    console.log("Aplicación cargada correctamente.");

    function marcarTabActiva() {
        const tabs = document.querySelectorAll(".nav-tabs .nav-link");
        tabs.forEach(tab => tab.classList.remove("tab-activa-extra"));

        const activa = document.querySelector(".nav-tabs .nav-link.active");
        if (activa) {
            activa.classList.add("tab-activa-extra");
        }
    }

    marcarTabActiva();

    document.addEventListener("click", function (e) {
        if (e.target.classList.contains("nav-link")) {
            setTimeout(marcarTabActiva, 100);
        }
    });

    document.addEventListener("click", function (e) {
        const texto = (e.target.innerText || "").trim().toLowerCase();
        if (texto === "generar predicción") {
            setTimeout(() => {
                const cards = document.querySelectorAll(".card-header");
                for (const c of cards) {
                    if ((c.innerText || "").toLowerCase().includes("serie histórica y predicción")) {
                        c.scrollIntoView({ behavior: "smooth", block: "start" });
                        break;
                    }
                }
            }, 700);
        }
    });

    // Ocultar botón del sidebar cuando se abra el datepicker
    document.addEventListener("focusin", function (e) {
        if (e.target.closest(".input-daterange")) {
            document.body.classList.add("datepicker-open");
        }
    });

    document.addEventListener("click", function (e) {
        const dentroDatepicker = e.target.closest(".datepicker") || e.target.closest(".input-daterange");
        if (!dentroDatepicker) {
            document.body.classList.remove("datepicker-open");
        }
    });

    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") {
            document.body.classList.remove("datepicker-open");
        }
    });
});