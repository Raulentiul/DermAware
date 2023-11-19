let map;
let service;
let infowindow;

async function initMap() {
  const { Map } = await google.maps.importLibrary("maps");
  const { AdvancedMarkerElement } = await google.maps.importLibrary("marker");
  const { PlacesService, PlacesServiceStatus } = await google.maps.importLibrary("places");

  const center = { lat: 45.760696, lng: 21.226788 };
  map = new Map(document.getElementById("map"), {
    zoom: 13,
    center,
    mapId: "4504f8b37365c3d0",
    disableDefaultUI: true,
  });

  const request = {
    query: ["dermatologie"],
    fields: ["name", "geometry"],
  };

  service = new PlacesService(map);

  service.textSearch(request, (results, status) => {
    if (status === PlacesServiceStatus.OK && results) {
      const properties = [];

      for (let i = 0; i < results.length; i++) {
        const result = results[i];
        const property = {
          address: result.formatted_address,
          type: "hospital", 
          position: result.geometry.location,
        };
        properties.push(property);
      }

      for (const property of properties) {
        const AdvancedMarkerElement = new google.maps.marker.AdvancedMarkerElement({
          map,
          content: buildContent(property),
          position: property.position,
          title: property.description,
        });

        AdvancedMarkerElement.addListener("click", () => {
          toggleHighlight(AdvancedMarkerElement, property);
        });
      }

      console.log(properties); 
    }
  });
}

function toggleHighlight(markerView) {
  if (markerView.content.classList.contains("highlight")) {
    markerView.content.classList.remove("highlight");
    markerView.zIndex = null;
  } else {
    markerView.content.classList.add("highlight");
    markerView.zIndex = 1;
  }
}

function buildContent(property) {
  const content = document.createElement("div");

  content.classList.add("property");
  content.innerHTML = `
    <div class="icon">
        <i aria-hidden="true" class="fa fa-icon fa-${property.type}" title="${property.type}"></i>
        <span class="fa-sr-only">${property.type}</span>
    </div>
    <div class="details">
        <div class="address">${property.address}</div>
        </div>
    </div>
    `;
  return content;
}

initMap();
