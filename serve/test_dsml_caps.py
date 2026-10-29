"""Native DSML capability discovery must use the supported DSML spelling."""
import unittest
from serve.frontend import ChatTemplate
from serve.dsml import FIXTURE_TEMPLATE

class DsmlCaps(unittest.TestCase):
    def test_native_template_advertises_its_rendered_tools(self):
        caps=ChatTemplate(FIXTURE_TEMPLATE).caps
        for name in ('supports_tools','supports_tool_calls','supports_parallel_tool_calls','supports_system_role'):
            self.assertTrue(caps[name],(name,caps))

if __name__=='__main__':unittest.main()
