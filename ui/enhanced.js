// UI Enhancements - Toast Notification System

(function(){
  // Ensure we only initialize once
  if (window.__illustrious_toast_init) return;
  window.__illustrious_toast_init = true;

  function createContainer(){
    let container = document.querySelector('.toast-container');
    if(!container){
      container = document.createElement('div');
      container.className = 'toast-container';
      document.body.appendChild(container);
    }
    return container;
  }

  window.showToast = function(message, type){
    type = type || 'info';
    const container = createContainer();
    const toast = document.createElement('div');
    toast.className = 'toast toast-' + type;
    toast.textContent = message;
    container.appendChild(toast);
    // Force reflow for animation
    void toast.offsetWidth;
    toast.classList.add('show');
    setTimeout(() => {
      toast.classList.remove('show');
      toast.addEventListener('transitionend', () => toast.remove(), {once: true});
    }, 3000);
  };

  window.notifyStatus = function(status){
    if(!status) return;
    let type = 'info';
    if(status.startsWith('✅')) type = 'success';
    else if(status.startsWith('❌')) type = 'error';
    window.showToast(status, type);
  };
})();
