// Figures and result tables are static; switching periods never recalculates research.
document.querySelectorAll('[data-period]').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-period]').forEach(item => {
      const active = item === button;
      item.classList.toggle('active', active);
      item.setAttribute('aria-pressed', String(active));
      document.getElementById(item.dataset.period).hidden = !active;
    });
    document.querySelector('#performance-table caption').textContent =
      '五档组合与中证1000绩效：' + button.textContent;
  });
});
