import { Header as SpectrumHeader, View, Flex } from '@geti/ui';

export const Header = () => {
  return (
    <View backgroundColor={'gray-300'} height="size-800">
      <Flex height="100%" alignItems={'center'} marginX="1rem" gap="size-200">
        <SpectrumHeader>Geti Prompt</SpectrumHeader>
      </Flex>
    </View>
  );
};
