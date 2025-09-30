/*
 * Copyright 2025 Thousand Brains Project
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT license
 * that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

const DOCS_BASE_URL = 'https://thousandbrainsproject.readme.io/docs/';
const GITHUB_EDIT_BASE_URL = 'https://github.com/thousandbrainsproject/tbp.monty/edit/main/';
const GITHUB_AVATAR_URL = 'https://github.com';
const EXTERNAL_LINK_ICON = 'fa-external-link-alt';
const EDIT_ICON = 'fa-pencil-alt';
const BADGE_CLASS = 'badge';
const BADGE_SKILLS_CLASS = 'badge-skills';


function escapeHtml(unsafe) {
  if (unsafe == null) return '';
  return he.encode(String(unsafe));
}


function addToSearch(value) {
  const input = document.getElementById('searchInput');
  const currentValue = input.value.trim();
  const searchTerm = value.trim();

  if (currentValue.includes(searchTerm)) {
    input.value = currentValue.replace(searchTerm, '').replace(/\s+/g, ' ').trim();
  } else {
    input.value = currentValue ? `${currentValue} ${searchTerm}` : searchTerm;
  }

  input.dispatchEvent(new Event('input', { bubbles: true }));
}


function updateUrlSearchParam(searchTerm) {
  const url = new URL(window.location);

  if (searchTerm.trim()) {
    url.searchParams.set('q', searchTerm.trim());
  } else {
    url.searchParams.delete('q');
  }

  window.history.replaceState({}, '', url);
}


function getInitialSearchTerm() {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get('q') || '';
}


const ColumnFormatters = {

  formatArrayOrStringColumn(value, cssClass) {
    const items = Array.isArray(value)
      ? value.filter(Boolean)
      : (value || '').split(',').map(item => item.trim()).filter(Boolean);

    return items
      .map(item => `<span class="${escapeHtml(cssClass)}" data-search-value="${escapeHtml(item)}" style="cursor: pointer;">${escapeHtml(item)}</span>`)
      .join(' ');
  },
  formatLinkColumn(cell, icon = EXTERNAL_LINK_ICON, urlPrefix = '') {
    const value = cell.getValue();
    if (!value) return '';

    const url = urlPrefix ? `${urlPrefix}${value}` : value;
    return `<a href="${url}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(url)}"><i class="${icon}"></i></a>`;
  },
  formatTagsColumn: (cell) => ColumnFormatters.formatArrayOrStringColumn(cell.getValue(), BADGE_CLASS),
  formatSkillsColumn: (cell) => ColumnFormatters.formatArrayOrStringColumn(cell.getValue(), BADGE_SKILLS_CLASS),
  formatSizeColumn(cell) {
    const value = (cell.getValue() || '').trim().toLowerCase();
    return value
      ? `<span class="badge badge-size-${escapeHtml(value)}" data-search-value="${escapeHtml(value)}" style="cursor: pointer;">${escapeHtml(value)}</span>`
      : '';
  },
  formatSlugLinkColumn: (cell) => ColumnFormatters.formatLinkColumn(cell, EXTERNAL_LINK_ICON, DOCS_BASE_URL),
  formatEditLinkColumn: (cell) => ColumnFormatters.formatLinkColumn(cell, EDIT_ICON, GITHUB_EDIT_BASE_URL),
  formatTitleWithLinksColumn(cell) {
    const rowData = cell.getRow().getData();
    const title = cell.getValue() || '';
    const slug = rowData.slug || '';
    const path = rowData.path || '';

    let result = escapeHtml(title);

    if (slug) {
      const docsUrl = `${DOCS_BASE_URL}${slug}`;
      result = `<a href="${escapeHtml(docsUrl)}" target="_blank" rel="noopener noreferrer" style="text-decoration: none; color: inherit;">${escapeHtml(title)}</a>`;
    }

    if (path) {
      const editUrl = `${GITHUB_EDIT_BASE_URL}${path}`;
      result = `<a href="${escapeHtml(editUrl)}" style="margin-right:5px;" target="_blank" rel="noopener noreferrer" title="Edit on GitHub"><i class="fas ${EDIT_ICON}"></i></a>${result}`;
    }

    return `<div style="margin-right: 10px;">${result}</div>`;
  },
  formatStatusColumn(cell) {
    const rowData = cell.getRow().getData();
    const status = cell.getValue() || '';
    const contributor = rowData.contributor || '';

    if (!contributor) return escapeHtml(status);

    const usernames = Array.isArray(contributor)
      ? contributor
      : contributor.split(',').map(u => u.trim()).filter(Boolean);

    const avatars = usernames
      .map(username => `<img src="${GITHUB_AVATAR_URL}/${encodeURIComponent(username)}.png"
                             width="16" height="16"
                             style="vertical-align:middle;border-radius:2px;margin-left:5px;"
                             alt="${escapeHtml(username)}"/>`)
      .join(' ');

    return escapeHtml(status) + avatars;
  },
  formatRfcColumn(cell) {
    const value = cell.getValue();
    if (!value) return '';

    const isHttpUrl = /^https?:/.test(value.trim());
    return isHttpUrl
      ? `<a href="${escapeHtml(value)}" target="_blank" rel="noopener noreferrer">RFC <i class="fas ${EXTERNAL_LINK_ICON}"></i></a>`
      : escapeHtml(value);
  }
};


const TableConfig = {

  getColumnsToShow() {
    const urlParams = new URLSearchParams(window.location.search);
    const columnsParam = urlParams.get('columns');

    return columnsParam
      ? columnsParam.split(',').map(col => col.trim().toLowerCase())
      : null;
  },


  getAllColumns() {
    return [
      { title: 'Title', field: 'title', formatter: ColumnFormatters.formatTitleWithLinksColumn },
      { title: 'Estimated Scope', field: 'estimated-scope', formatter: ColumnFormatters.formatSizeColumn },
      { title: 'RFC', field: 'rfc', formatter: ColumnFormatters.formatRfcColumn },
      { title: 'Status', field: 'status', formatter: ColumnFormatters.formatStatusColumn },
      { title: 'Tags', field: 'tags', formatter: ColumnFormatters.formatTagsColumn, widthGrow: 2, cssClass: 'wrap-text' },
      { title: 'Skills', field: 'skills', formatter: ColumnFormatters.formatSkillsColumn, widthGrow: 2, cssClass: 'wrap-text' }
    ];
  },


  getDisplayColumns() {
    const allColumns = this.getAllColumns();
    const columnsToShow = this.getColumnsToShow();

    return columnsToShow
      ? allColumns.filter(col =>
          columnsToShow.includes(col.field.toLowerCase())
        )
      : allColumns;
  }
};


const FutureWorkWidget = {

  async init() {
    try {
      const data = await this.loadData();
      const table = this.createTable(data);
      this.setupSearch(table);
    } catch (error) {
      console.error('Failed to initialize Future Work Widget:', error);
      this.showError('Failed to load data - see the console for more details or refresh the page to try again.');
    }
  },


  async loadData() {
    const response = await fetch('data.json');
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}, body: ${await response.text()}`);
    }
    const data = await response.json();
    return Array.isArray(data)
      ? data.slice().sort((a, b) =>
          (a.path2 || '').localeCompare(b.path2 || '', undefined, { sensitivity: 'base' }) ||
          (a.title || '').localeCompare(b.title || '', undefined, { sensitivity: 'base' })
        )
      : data;
  },


  createTable(data) {
    return new Tabulator('#table', {
      data: data,
      layout: 'fitDataStretch',
      columns: TableConfig.getDisplayColumns(),
      groupBy: 'path2'
    });
  },


  setupSearch(table) {
    const searchInput = document.getElementById('searchInput');
    if (!searchInput) {
      console.warn('Search input not found');
      return;
    }

    const initialSearchTerm = getInitialSearchTerm();
    if (initialSearchTerm) {
      searchInput.value = initialSearchTerm;
    }

    const performSearch = (searchTerm) => {
      const trimmedTerm = searchTerm.toLowerCase().trim();

      if (!trimmedTerm) {
        table.clearFilter();
        return;
      }

      table.setFilter((data) => {
        const searchableText = [
          data.title, data.tags, data.skills, data.status,
          data.contributor, data['estimated-scope'], data.rfc, data.link, data.path2
        ]
          .filter(Boolean)
          .join(' ')
          .toLowerCase();

        return trimmedTerm
          .split(/\s+/)
          .every(word => searchableText.includes(word));
      });
    };

    if (initialSearchTerm) {
      performSearch(initialSearchTerm);
    }

    searchInput.addEventListener('input', (e) => {
      const searchTerm = e.target.value;
      updateUrlSearchParam(searchTerm);
      performSearch(searchTerm);
    });

    const clearLink = document.getElementById('clearSearch');
    if (clearLink) {
      clearLink.addEventListener('click', (e) => {
        e.preventDefault();
        searchInput.value = '';
        updateUrlSearchParam('');
        table.clearFilter();
      });
    }

    const copyUrlLink = document.getElementById('copyUrl');
    if (copyUrlLink) {
      copyUrlLink.addEventListener('click', async (e) => {
        e.preventDefault();
        try {
          await navigator.clipboard.writeText(window.location.href);
          copyUrlLink.textContent = '✅';
          copyUrlLink.classList.add('success');
          setTimeout(() => {
            copyUrlLink.textContent = '📋';
            copyUrlLink.classList.remove('success');
          }, 1500);
        } catch (err) {
          console.error('Failed to copy URL to clipboard:', err);
          copyUrlLink.textContent = '❌';
          setTimeout(() => {
            copyUrlLink.textContent = '📋';
          }, 1500);
        }
      });
    }

    document.addEventListener('click', (e) => {
      const searchValue = e.target.dataset.searchValue;
      if (searchValue) {
        addToSearch(searchValue);
      }
    });
  },


  showError(message) {
    const errorElement = document.createElement('div');
    errorElement.className = 'error-message';
    errorElement.textContent = message;
    errorElement.style.cssText = 'color: red; padding: 20px; text-align: center; font-weight: bold;';

    const tableElement = document.getElementById('table');
    if (tableElement) {
      tableElement.appendChild(errorElement);
    }
  }
};


document.addEventListener('DOMContentLoaded', () => {
  FutureWorkWidget.init();
});
