package plugin.editor;

import java.util.MissingResourceException;
import java.util.ResourceBundle;

public class CorpusEditorMessages {

	private static final String RESOURCE_BUNDLE = CorpusEditorMessages.class.getName();
	private static ResourceBundle fgResourceBundle= ResourceBundle.getBundle(RESOURCE_BUNDLE);

	private CorpusEditorMessages() {
	}

	public static String getString(String key) {
		try {
			return fgResourceBundle.getString(key);
		} catch (MissingResourceException e) {
			return "!" + key + "!";//$NON-NLS-2$ //$NON-NLS-1$
		}
	}

	public static ResourceBundle getResourceBundle() {
		return fgResourceBundle;
	}
}
