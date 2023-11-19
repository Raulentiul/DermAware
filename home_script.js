 <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/js/all.min.js"></script>
    <script>
        function scrollToMain() {
            const mainSection = document.getElementById('main');
            mainSection.scrollIntoView({ behavior: 'smooth' });
        }

        window.addEventListener('scroll', function() {
            const bannerOverlay = document.querySelector('.banner-overlay');
            const bannerHeight = document.querySelector('.banner').clientHeight;
            const scrolledHeight = window.pageYOffset;
            const opacity = 1 - (scrolledHeight / bannerHeight);
            bannerOverlay.style.opacity = opacity;
            if (opacity <= 0) {
                bannerOverlay.classList.add('hide');
            } else {
                bannerOverlay.classList.remove('hide');
            }
        });

        document.addEventListener('DOMContentLoaded', function() {
            var alertElement = document.getElementById('myAlert');

            alertElement.addEventListener('click', function() {
                alertElement.style.display = 'none';
            });
        });
    </script>