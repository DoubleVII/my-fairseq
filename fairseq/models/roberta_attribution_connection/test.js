var autoclicker = setInterval(function () {
    try {
        document.getElementById('bigCookie').click();
    } catch (err) {
        clearInterval();
    }
}, 100);


var auto_golden_clicker = setInterval(function () {
    try {
        shimmers = document.getElementsByClassName("shimmer");
        if (shimmers.length > 0) {
            shimmers[0].click();
        }
    } catch (err) {
        clearInterval();
    }
}, 100);


clearInterval(autoclicker);

clearInterval(auto_golden_clicker);


shimmers = document.getElementsByClassName("shimmer");
shimmer_div = document.getElementsById("shimmers");
copy_golden = shimmers[0].cloneNode(true)
shimmer_div = document.getElementsById("shimmers");
shimmer_div.appendChild(copy_golden)