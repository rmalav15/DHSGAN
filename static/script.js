$(function () {
    var $dehaze = $('#dehaze');
    var $real = $('#real');
    var $tmap = $('#tmap');
    var $result = $('#result');

    // Activate tooltip
    $('[data-toggle="tooltip"]').tooltip();

    $dehaze.on('click', function (e) {
        e.preventDefault();
        console.log('report');
        dehazeImage();
    });


    function dehazeImage() {
        var form = document.querySelector('#image-form');
        var formData = new FormData(form);
        $.ajax({
            url: "/dehaze",
            data: formData,
            type: 'POST',
            contentType: false,
            cache: false,
            processData: false,
            async: false,
            success: function (response) {
                console.log(response['real']);
                $real.attr("src",  response['real']);
                $tmap.attr("src",  response['tmap']);
                $result.attr("src", response['dehazed']);
                return true;
            }
        });
    }

});
